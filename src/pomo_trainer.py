"""POMO pre-training loop (REINFORCE with multi-start baseline)."""

import gc
import ctypes
import logging
import random as pyrandom
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.pomo_model import POMOModel
from src.pomo_env import VRPTWEnvironment, VRPTWState
from src.utils import normalize_coordinates

logger = logging.getLogger(__name__)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _get_system_memory_load_percent() -> float:
    try:
        stat = _MemoryStatus()
        stat.dwLength = ctypes.sizeof(_MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return float(stat.dwMemoryLoad)
    except Exception:
        return float("nan")


# ======================================================================
# Random instance generation (training data)
# ======================================================================

class InstanceGenerator:
    """Produce batched VRPTW instances for training.

    Supports:
      - synthetic fallback
      - single-map real sampling
      - mixed-map real sampling with stage-aware curriculum
      - optional 8-fold coordinate augmentation
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device,
        problem=None,
        problems: Optional[List] = None,
        instance_specs: Optional[List] = None,
    ) -> None:
        self.device = device
        self.config = config
        self.horizon = config["problem"].get("planning_horizon_hours", 24.0)
        self.training_data_config = config.get("training_data", {})
        self.augment_8fold = bool(self.training_data_config.get("augment_8fold", False))
        self.real_data_mix_ratio = float(self.training_data_config.get("real_data_mix_ratio", 0.75))
        self.synthetic_data_mix_ratio = float(
            self.training_data_config.get("synthetic_data_mix_ratio", 0.25)
        )

        all_problems: List = []
        if problems is not None:
            all_problems.extend(problems)
        if problem is not None:
            all_problems.append(problem)

        self.problem_records: List[Dict] = []
        for idx, loaded_problem in enumerate(all_problems):
            spec = instance_specs[idx] if instance_specs and idx < len(instance_specs) else None
            record = {
                "problem": loaded_problem,
                "spec": spec,
                "vehicle_data": self._build_vehicle_data(loaded_problem),
                "full_dist_km": loaded_problem.distance_matrix_meters / 1000.0,
                "customer_count": int(loaded_problem.num_customers),
                "instance_name": getattr(spec, "instance_name", f"problem_{idx}"),
            }
            self.problem_records.append(record)

        self.stage_threshold_tiny = int(self.training_data_config.get("stage_tiny_max_customers", 8))
        self.stage_threshold_medium = int(self.training_data_config.get("stage_medium_max_customers", 40))
        self.current_stage = "main"

        self.stage_weights = self.training_data_config.get(
            "stage_sampling_weights",
            {
                "tiny_warmup": {"tiny": 1.0, "small_medium": 0.0, "large": 0.0},
                "main": {"tiny": 0.1, "small_medium": 0.45, "large": 0.45},
            },
        )

    # ------------------------------------------------------------------
    # Stage control
    # ------------------------------------------------------------------

    def set_stage(self, stage_name: str) -> None:
        self.current_stage = stage_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self, batch_size: int, num_customers: int,
    ) -> Dict:
        if self.problem_records and self._should_sample_real_data():
            return self._generate_from_mixed_real_data(batch_size, num_customers)
        return self._generate_synthetic(batch_size, num_customers)

    def _should_sample_real_data(self) -> bool:
        real_ratio = max(0.0, self.real_data_mix_ratio)
        synthetic_ratio = max(0.0, self.synthetic_data_mix_ratio)
        total_ratio = real_ratio + synthetic_ratio
        if total_ratio <= 1e-10:
            logger.warning(
                "Invalid training_data mix ratios (real=%s, synthetic=%s). "
                "Falling back to real-only sampling.",
                self.real_data_mix_ratio,
                self.synthetic_data_mix_ratio,
            )
            return True
        real_probability = real_ratio / total_ratio
        return pyrandom.random() < real_probability

    # ------------------------------------------------------------------
    # Real-data sub-graph sampling
    # ------------------------------------------------------------------

    def _build_vehicle_data(self, problem) -> Dict[str, Dict]:
        vehicle_data: Dict[str, Dict] = {}
        for v_idx, vtype in enumerate(problem.vehicle_types):
            accessible = [
                c for c in range(problem.num_customers)
                if problem.site_dependency[c, v_idx]
            ]
            if accessible:
                vehicle_data[vtype] = {
                    "accessible": accessible,
                    "capacity": problem.vehicle_capacity[vtype],
                    "fixed_cost": problem.vehicle_fixed_cost[vtype],
                    "cost_per_km": problem.vehicle_cost_per_km[vtype],
                    "cost_per_hour": problem.vehicle_cost_per_hour[vtype],
                    "speed_kmh": problem.vehicle_speed_kmh[vtype],
                    "v_idx": v_idx,
                }
        return vehicle_data

    def _bucket_for_record(self, record: Dict) -> str:
        count = record["customer_count"]
        if count <= self.stage_threshold_tiny:
            return "tiny"
        if count <= self.stage_threshold_medium:
            return "small_medium"
        return "large"

    def _choose_problem_record(self) -> Dict:
        if len(self.problem_records) == 1:
            return self.problem_records[0]

        groups = {"tiny": [], "small_medium": [], "large": []}
        for record in self.problem_records:
            groups[self._bucket_for_record(record)].append(record)

        stage_key = self.current_stage if self.current_stage in self.stage_weights else "main"
        raw_weights = self.stage_weights.get(stage_key, self.stage_weights.get("main", {}))

        available_groups: List[str] = []
        available_weights: List[float] = []
        for group_name, group_records in groups.items():
            if not group_records:
                continue
            available_groups.append(group_name)
            available_weights.append(float(raw_weights.get(group_name, 0.0)))

        if not available_groups:
            return pyrandom.choice(self.problem_records)

        if sum(available_weights) <= 1e-10:
            chosen_group = pyrandom.choice(available_groups)
        else:
            chosen_group = pyrandom.choices(
                population=available_groups,
                weights=available_weights,
                k=1,
            )[0]
        return pyrandom.choice(groups[chosen_group])

    def _augment_xy_data_by_8_fold(self, xy_data: torch.Tensor) -> torch.Tensor:
        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)
        return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)

    def _repeat_batch_for_augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat_interleave(8, dim=0)

    def _generate_from_mixed_real_data(
        self, batch_size: int, num_customers: int,
    ) -> Dict:
        record = self._choose_problem_record()
        problem = record["problem"]
        full_dist_km = record["full_dist_km"]
        vehicle_data = record["vehicle_data"]
        dev = self.device

        if not vehicle_data:
            return self._generate_synthetic(batch_size, num_customers)

        chosen_vehicle_type = pyrandom.choice(list(vehicle_data.keys()))
        chosen_vehicle_data = vehicle_data[chosen_vehicle_type]
        accessible_customers = chosen_vehicle_data["accessible"]
        capacity = chosen_vehicle_data["capacity"]

        num_sampled_customers = min(num_customers, len(accessible_customers))
        num_nodes = num_sampled_customers + 1

        coords_np = np.zeros((batch_size, num_nodes, 2), dtype=np.float64)
        demands_np = np.zeros((batch_size, num_nodes), dtype=np.float64)
        tw_s_np = np.zeros((batch_size, num_nodes), dtype=np.float64)
        tw_e_np = np.full((batch_size, num_nodes), self.horizon, dtype=np.float64)
        svc_np = np.zeros((batch_size, num_nodes), dtype=np.float64)
        dist_np = np.zeros((batch_size, num_nodes, num_nodes), dtype=np.float64)
        tt_np = np.zeros((batch_size, num_nodes, num_nodes), dtype=np.float64)

        full_tt = problem.travel_time_matrices[chosen_vehicle_type]

        for batch_index in range(batch_size):
            sampled_customers = pyrandom.sample(accessible_customers, num_sampled_customers)
            full_indices = [0] + [c + 1 for c in sampled_customers]
            ix = np.ix_(full_indices, full_indices)

            sub_coords = problem.all_coords[full_indices]
            coords_np[batch_index] = normalize_coordinates(sub_coords.copy())

            for local_idx, customer_idx in enumerate(sampled_customers):
                demands_np[batch_index, local_idx + 1] = problem.demands[customer_idx]
                tw_s_np[batch_index, local_idx + 1] = problem.tw_start[customer_idx]
                tw_e_np[batch_index, local_idx + 1] = problem.tw_end[customer_idx]
                svc_np[batch_index, local_idx + 1] = problem.service_times[customer_idx]

            tw_s_np[batch_index, 0] = problem.depot_tw_start
            tw_e_np[batch_index, 0] = problem.depot_tw_end
            dist_np[batch_index] = full_dist_km[ix]
            tt_np[batch_index] = full_tt[ix]

        max_cost_est = (
            chosen_vehicle_data["fixed_cost"]
            + chosen_vehicle_data["cost_per_km"] * 100
            + chosen_vehicle_data["cost_per_hour"] * 5
        )
        dual_np = np.random.rand(batch_size, num_nodes) * max_cost_est * 0.5
        dual_np[:, 0] = 0.0

        site_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=dev)

        batch = {
            "coords": torch.tensor(coords_np, dtype=torch.float32, device=dev),
            "demands": torch.tensor(demands_np, dtype=torch.float32, device=dev),
            "tw_start": torch.tensor(tw_s_np, dtype=torch.float32, device=dev),
            "tw_end": torch.tensor(tw_e_np, dtype=torch.float32, device=dev),
            "service_times": torch.tensor(svc_np, dtype=torch.float32, device=dev),
            "distance_km": torch.tensor(dist_np, dtype=torch.float32, device=dev),
            "travel_time": torch.tensor(tt_np, dtype=torch.float32, device=dev),
            "dual_values": torch.tensor(dual_np, dtype=torch.float32, device=dev),
            "site_mask": site_mask,
            "vehicle_capacity": capacity,
            "fixed_cost": chosen_vehicle_data["fixed_cost"],
            "cost_per_km": chosen_vehicle_data["cost_per_km"],
            "cost_per_hour": chosen_vehicle_data["cost_per_hour"],
            "instance_name": record["instance_name"],
        }

        if self.augment_8fold:
            batch["coords"] = self._augment_xy_data_by_8_fold(batch["coords"])
            for key in ["demands", "tw_start", "tw_end", "service_times", "distance_km", "travel_time", "dual_values", "site_mask"]:
                batch[key] = self._repeat_batch_for_augmentation(batch[key])
            expected_batch = batch_size * 8
            if batch["coords"].shape[0] != expected_batch:
                raise ValueError(
                    f"8-fold augmentation shape mismatch: expected B={expected_batch}, "
                    f"got B={batch['coords'].shape[0]}"
                )

        return batch

    # ------------------------------------------------------------------
    # Synthetic fallback (improved cost alignment)
    # ------------------------------------------------------------------

    def _generate_synthetic(
        self, batch_size: int, num_customers: int,
    ) -> Dict:
        dev = self.device
        num_nodes = num_customers + 1
        H = self.horizon

        coords = torch.rand(batch_size, num_nodes, 2, device=dev)

        demands = torch.zeros(batch_size, num_nodes, device=dev)
        demands[:, 1:] = torch.randint(
            1, 51, (batch_size, num_customers),
            device=dev, dtype=torch.float32,
        )

        tw_start = torch.zeros(batch_size, num_nodes, device=dev)
        tw_end = torch.full((batch_size, num_nodes), H, device=dev)
        rand_start = torch.rand(batch_size, num_customers, device=dev) * (H - 6)
        window_w = torch.rand(batch_size, num_customers, device=dev) * 4 + 2
        tw_start[:, 1:] = rand_start
        tw_end[:, 1:] = (rand_start + window_w).clamp(max=H)

        service = torch.zeros(batch_size, num_nodes, device=dev)
        service[:, 1:] = (
            torch.rand(batch_size, num_customers, device=dev) * 0.8 + 0.2
        )

        scale_km = 50.0
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distance_km = diff.pow(2).sum(dim=-1).sqrt() * scale_km
        travel_time = distance_km / 30.0

        max_expected_dual = 150.0
        dual_values = torch.rand(batch_size, num_nodes, device=dev) * max_expected_dual
        dual_values[:, 0] = 0.0
        site_mask = torch.ones(
            batch_size, num_nodes, dtype=torch.bool, device=dev,
        )

        return {
            "coords": coords,
            "demands": demands,
            "tw_start": tw_start,
            "tw_end": tw_end,
            "service_times": service,
            "distance_km": distance_km,
            "travel_time": travel_time,
            "dual_values": dual_values,
            "site_mask": site_mask,
            "vehicle_capacity": 2000.0,
            "fixed_cost": 15.0,
            "cost_per_km": 12.0,
            "cost_per_hour": 3.0,
        }


# ======================================================================
# Feature builder (shared with pricing)
# ======================================================================

def build_node_features(
    instance: Dict[str, torch.Tensor],
    max_capacity: float,
    max_time: float,
) -> torch.Tensor:
    """Construct the 7-dimensional node feature tensor.

    Features per node:
        [x, y, demand/capacity, tw_start/horizon,
         tw_end/horizon, service/horizon, dual_value]
    """
    c = instance["coords"]                         # (B, N, 2)
    d = instance["demands"] / max_capacity
    ts = instance["tw_start"] / max_time
    te = instance["tw_end"] / max_time
    sv = instance["service_times"] / max_time
    du = instance["dual_values"]
    du_max = du.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    du_norm = du / du_max

    return torch.stack(
        [c[:, :, 0], c[:, :, 1], d, ts, te, sv, du_norm], dim=-1
    )                                              # (B, N, 7)


# ======================================================================
# POMO rollout
# ======================================================================

def run_pomo_rollout(
    model: POMOModel,
    env: VRPTWEnvironment,
    instance: Dict[str, torch.Tensor],
    num_pomo_starts: int,
    vehicle_capacity: float,
    depot_tw_end: float,
    decode_method: str = "sampling",
    temperature: float = 1.0,
    fixed_cost: float = 10.0,
    cost_per_km: float = 1.0,
    cost_per_hour: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Execute a full POMO rollout for a batch of instances.

    Returns dict with keys: costs, log_probs, distances, times, sequences
        each shaped (B_orig, P).
    """
    device = next(model.parameters()).device
    B_orig = instance["coords"].shape[0]
    N = instance["coords"].shape[1]
    num_customers = N - 1
    P = min(num_pomo_starts, num_customers)
    B = B_orig * P

    # Replicate along the POMO dimension
    def rep(t: torch.Tensor) -> torch.Tensor:
        return (
            t.unsqueeze(1)
            .expand(-1, P, *[-1] * (t.dim() - 1))
            .reshape(B, *t.shape[1:])
        )

    node_feat = build_node_features(instance, vehicle_capacity, depot_tw_end)
    node_feat_r = rep(node_feat)

    model.set_decode_context(vehicle_capacity, depot_tw_end)
    enc_out, graph_emb = model.encode(node_feat_r)

    state = env.reset(
        demands=rep(instance["demands"]),
        coords=rep(instance["coords"]),
        tw_start=rep(instance["tw_start"]),
        tw_end=rep(instance["tw_end"]),
        service_times=rep(instance["service_times"]),
        travel_time_matrix=rep(instance["travel_time"]),
        distance_matrix_km=rep(instance["distance_km"]),
        vehicle_capacity=vehicle_capacity,
        depot_tw_end=depot_tw_end,
        site_mask=rep(instance["site_mask"]),
    )

    # First step — force distinct starting customers for POMO diversity
    pomo_starts = (
        torch.arange(1, P + 1, device=device)
        .unsqueeze(0)
        .expand(B_orig, -1)
        .reshape(B)
    )

    mask = env.get_action_mask(state)
    forced_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    forced_mask[torch.arange(B, device=device), pomo_starts] = False

    forced_infeasible = mask[torch.arange(B, device=device), pomo_starts]
    active_mask = torch.where(forced_infeasible.unsqueeze(1), mask, forced_mask)

    logits = model.decoder(
        enc_out, graph_emb,
        state.first_customer, state.current_node,
        (state.remaining_capacity / (vehicle_capacity + 1e-8)).clamp(0, 1),
        (state.current_time / (depot_tw_end + 1e-8)).clamp(0, 1),
        active_mask,
    )
    probs = F.softmax(logits, dim=-1)
    action = torch.where(forced_infeasible, logits.argmax(dim=-1), pomo_starts)
    
    # 3. The Masking Assertion Trap
    assert not active_mask[torch.arange(B, device=device), action].any(), "FATAL: POMO selected a masked node!"
    
    log_prob = torch.log(probs.gather(1, action.unsqueeze(1)).squeeze(1) + 1e-8)
    state = env.step(state, action, log_prob)

    for _ in range(N + 10):
        if env.is_done(state):
            break
        mask = env.get_action_mask(state)
        action, log_prob = model.decode_step(
            enc_out, graph_emb, state, mask,
            decode_method=decode_method, temperature=temperature,
        )
        
        # 3. The Masking Assertion Trap
        assert not mask[torch.arange(B, device=device), action].any(), "FATAL: POMO selected a masked node!"
        
        state = env.step(state, action, log_prob)

    # Force depot return for any unfinished routes
    if not env.is_done(state):
        state = env.step(
            state,
            torch.zeros(B, dtype=torch.long, device=device),
            torch.zeros(B, dtype=torch.float32, device=device),
        )

    total_lp = torch.stack(state.log_probs, dim=1).sum(dim=1).view(B_orig, P)
    dists = state.total_distance_km.view(B_orig, P)
    times = state.total_time_hours.view(B_orig, P)
    
    route_costs = fixed_cost + dists * cost_per_km + times * cost_per_hour

    # Gather dual values for visited customers
    seqs = torch.stack(state.sequences, dim=1).view(B_orig, P, -1)
    dual_values_expanded = instance["dual_values"].unsqueeze(1).expand(B_orig, P, N)
    sum_duals = dual_values_expanded.gather(2, seqs).sum(dim=2)

    reduced_costs = route_costs - sum_duals

    # Add penalty for getting stuck
    arrival_at_depot = state.current_time.view(B_orig, P)
    is_stuck = arrival_at_depot > depot_tw_end + 1e-5
    penalty = 1e6
    reduced_costs = torch.where(is_stuck, reduced_costs + penalty, reduced_costs)

    rewards = -reduced_costs
    
    # Track penalty ratio
    penalty_ratio = is_stuck.float().mean().item()

    return {
        "rewards": rewards,
        "reduced_costs": reduced_costs,
        "log_probs": total_lp,
        "distances": dists,
        "times": times,
        "sequences": seqs,
        "penalty_ratio": penalty_ratio
    }


# ======================================================================
# Training epoch
# ======================================================================

def train_epoch(
    model: POMOModel,
    optimizer: optim.Optimizer,
    instance_generator: InstanceGenerator,
    config: Dict,
    epoch: int,
) -> Dict[str, float]:
    """One training epoch. Returns metrics dict."""
    model.train()
    device = next(model.parameters()).device

    batch_size = config["training"]["batch_size"]
    num_instances = config["training"]["num_train_instances"]
    num_customers = config["training"]["max_customers_per_instance"]
    num_pomo = config["pomo"]["num_pomo_starts"]
    vehicle_cap = 1000.0
    depot_tw_end = config["problem"].get("planning_horizon_hours", 24.0)

    env = VRPTWEnvironment(device)
    total_loss = 0.0
    total_reward = 0.0
    total_variance = 0.0
    total_penalty_ratio = 0.0
    max_memory_load_percent = float("nan")
    num_batches = max(1, num_instances // batch_size)
    epoch_start_time = time.time()

    for b_idx in range(num_batches):
        instance = instance_generator.generate_batch(batch_size, num_customers)

        batch_cap = instance.get("vehicle_capacity", vehicle_cap)
        batch_fixed = instance.get("fixed_cost", 10.0)
        batch_cpkm = instance.get("cost_per_km", 1.0)
        batch_cphr = instance.get("cost_per_hour", 1.0)

        result = run_pomo_rollout(
            model, env, instance, num_pomo,
            batch_cap, depot_tw_end,
            decode_method="sampling",
            fixed_cost=batch_fixed,
            cost_per_km=batch_cpkm,
            cost_per_hour=batch_cphr,
        )

        rewards = result["rewards"]        # (B, P)
        log_probs = result["log_probs"]

        # Identify valid rollouts (assuming penalty is -1e6)
        # We use -1e5 as a safe threshold
        valid_mask = rewards > -1e5 
        
        # Calculate mean only over valid rollouts to prevent baseline poisoning
        # If all rollouts in a batch item are invalid, fallback to the unmasked mean to avoid division by zero
        sum_valid = (rewards * valid_mask).sum(dim=1)
        count_valid = valid_mask.sum(dim=1)
        
        baseline = torch.where(
            count_valid > 0,
            sum_valid / count_valid.clamp(min=1),
            rewards.mean(dim=1)
        ).unsqueeze(1)
        
        advantage = rewards - baseline

        # REINFORCE: maximize reward -> minimize - (R - b) * log_prob
        loss = -(advantage.detach() * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_reward += rewards.mean().item()
        total_variance += rewards.var(dim=1, unbiased=False).mean().item()
        total_penalty_ratio += result["penalty_ratio"]
        memory_load = _get_system_memory_load_percent()
        if not np.isnan(memory_load):
            if np.isnan(max_memory_load_percent):
                max_memory_load_percent = memory_load
            else:
                max_memory_load_percent = max(max_memory_load_percent, memory_load)

        if (b_idx + 1) % 10 == 0:
            completed_batches = b_idx + 1
            progress = completed_batches / max(1, num_batches)
            elapsed_seconds = time.time() - epoch_start_time
            eta_seconds = (elapsed_seconds / progress) - elapsed_seconds if progress > 0 else float("nan")
            logger.info(
                "  batch %d/%d (%.1f%%)  loss=%.4f  avg_reward=%.2f  penalty_ratio=%.2f%%  eta=%.1fs",
                completed_batches,
                num_batches,
                progress * 100.0,
                loss.item(),
                rewards.mean().item(),
                result["penalty_ratio"] * 100,
                eta_seconds,
            )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss": total_loss / num_batches,
        "reward": total_reward / num_batches,
        "variance": total_variance / num_batches,
        "penalty_ratio": total_penalty_ratio / num_batches,
        "max_memory_load_percent": max_memory_load_percent,
    }
