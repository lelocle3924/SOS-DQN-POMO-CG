"""Pricing sub-problem: POMO rollouts per vehicle type to find negative-RC columns."""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data_loader import ProblemData
from src.pomo_model import POMOModel
from src.pomo_env import VRPTWEnvironment
from src.column_pool import ColumnPool, Route
from src.pomo_trainer import build_node_features
from src.utils import normalize_coordinates

logger = logging.getLogger(__name__)


def solve_pricing_for_all_vehicles(
    problem: ProblemData,
    model: POMOModel,
    dual_values: np.ndarray,
    column_pool: ColumnPool,
    config: Dict,
    forbidden_arcs: Optional[Set[Tuple[int, int]]] = None,
    enforced_arcs: Optional[Set[Tuple[int, int]]] = None,
) -> List[Route]:
    """Run pricing for every vehicle type.

    Returns the list of *newly added* columns with negative reduced cost.
    """
    device = next(model.parameters()).device
    new_routes: List[Route] = []

    for v_idx, vtype in enumerate(problem.vehicle_types):
        accessible = [
            c for c in range(problem.num_customers)
            if problem.site_dependency[c, v_idx]
        ]
        if not accessible:
            continue

        generated = _price_single_vehicle_type(
            problem, model, dual_values, vtype, v_idx,
            accessible, config, device,
            forbidden_arcs, enforced_arcs,
        )

        rc_tol = config["column_generation"]["reduced_cost_tolerance"]
        for cust_idx, cost, dist, time_h, rc in generated:
            if rc < rc_tol:
                added = column_pool.add_route(
                    vehicle_type=vtype,
                    customer_indices=cust_idx,
                    total_cost=cost,
                    total_distance_km=dist,
                    total_time_hours=time_h,
                    reduced_cost=rc,
                )
                if added is not None:
                    new_routes.append(added)
                    logger.debug(
                        "  +col %s  cust=%s  rc=%.4f",
                        vtype, cust_idx, rc,
                    )

    return new_routes


# ------------------------------------------------------------------
# Per-vehicle-type pricing
# ------------------------------------------------------------------

def _price_single_vehicle_type(
    problem: ProblemData,
    model: POMOModel,
    dual_values: np.ndarray,
    vehicle_type: str,
    vehicle_idx: int,
    accessible_customers: List[int],
    config: Dict,
    device: torch.device,
    forbidden_arcs: Optional[Set[Tuple[int, int]]],
    enforced_arcs: Optional[Set[Tuple[int, int]]],
) -> List[Tuple[List[int], float, float, float, float]]:
    """POMO rollout for one vehicle type.

    Returns list of (customer_indices, route_cost, dist_km, time_h, reduced_cost).
    """
    model.eval()
    num_nodes = problem.num_customers + 1
    capacity = problem.vehicle_capacity[vehicle_type]
    fixed_cost = problem.vehicle_fixed_cost[vehicle_type]
    cost_per_km = problem.vehicle_cost_per_km[vehicle_type]
    cost_per_hour = problem.vehicle_cost_per_hour[vehicle_type]
    depot_tw_end = problem.depot_tw_end
    num_pomo = config["pomo"]["num_pomo_starts"]

    instance = _build_instance_tensors(
        problem, dual_values, vehicle_type, vehicle_idx, device,
    )

    # Arc-constraint tensors
    forbidden_t = _build_arc_tensor(forbidden_arcs, num_nodes, device)
    enforced_t = _build_arc_tensor(enforced_arcs, num_nodes, device)

    P = min(num_pomo, len(accessible_customers))
    if P == 0:
        return []

    B = P  # single instance replicated P times

    def rep(t: torch.Tensor) -> torch.Tensor:
        return t.expand(B, *[-1] * (t.dim() - 1)).contiguous()

    node_feat = build_node_features(instance, capacity, depot_tw_end)
    model.set_decode_context(capacity, depot_tw_end)
    env = VRPTWEnvironment(device)

    with torch.no_grad():
        enc_out, graph_emb = model.encode(rep(node_feat))

        state = env.reset(
            demands=rep(instance["demands"]),
            coords=rep(instance["coords"]),
            tw_start=rep(instance["tw_start"]),
            tw_end=rep(instance["tw_end"]),
            service_times=rep(instance["service_times"]),
            travel_time_matrix=rep(instance["travel_time"]),
            distance_matrix_km=rep(instance["distance_km"]),
            vehicle_capacity=capacity,
            depot_tw_end=depot_tw_end,
            site_mask=rep(instance["site_mask"]),
            forbidden_arcs=rep(forbidden_t) if forbidden_t is not None else None,
            enforced_arcs=rep(enforced_t) if enforced_t is not None else None,
        )

        # Force distinct first customers for POMO diversity
        pomo_starts = torch.tensor(
            [c + 1 for c in accessible_customers[:P]],
            dtype=torch.long, device=device,
        )

        mask = env.get_action_mask(state)
        forced_mask = torch.ones(B, num_nodes, dtype=torch.bool, device=device)
        forced_mask[torch.arange(B, device=device), pomo_starts] = False
        forced_bad = mask[torch.arange(B, device=device), pomo_starts]
        active_mask = torch.where(forced_bad.unsqueeze(1), mask, forced_mask)

        logits = model.decoder(
            enc_out, graph_emb,
            state.first_customer, state.current_node,
            (state.remaining_capacity / (capacity + 1e-8)).clamp(0, 1),
            (state.current_time / (depot_tw_end + 1e-8)).clamp(0, 1),
            active_mask,
        )
        action = torch.where(forced_bad, logits.argmax(dim=-1), pomo_starts)
        zero_lp = torch.zeros(B, dtype=torch.float32, device=device)
        state = env.step(state, action, zero_lp)

        for _ in range(num_nodes + 10):
            if env.is_done(state):
                break
            mask = env.get_action_mask(state)
            action, lp = model.decode_step(
                enc_out, graph_emb, state, mask,
                decode_method="greedy",
            )
            state = env.step(state, action, lp)

        if not env.is_done(state):
            state = env.step(
                state,
                torch.zeros(B, dtype=torch.long, device=device),
                torch.zeros(B, dtype=torch.float32, device=device),
            )

    seqs = torch.stack(state.sequences, dim=1).cpu().numpy()
    dists = state.total_distance_km.cpu().numpy()
    times = state.total_time_hours.cpu().numpy()

    results = []
    for p in range(P):
        cust_indices = [int(n) - 1 for n in seqs[p] if n > 0]
        if not cust_indices:
            continue
        dist_km = float(dists[p])
        time_h = float(times[p])
        route_cost = fixed_cost + cost_per_km * dist_km + cost_per_hour * time_h
        dual_sum = sum(dual_values[c] for c in cust_indices)
        reduced_cost = route_cost - dual_sum
        results.append((cust_indices, route_cost, dist_km, time_h, reduced_cost))

    return results


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_instance_tensors(
    problem: ProblemData,
    dual_values: np.ndarray,
    vehicle_type: str,
    vehicle_idx: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create a single-instance (B=1) tensor dictionary for the pricing env."""
    N = problem.num_customers + 1

    coords_norm = normalize_coordinates(problem.all_coords.copy())
    coords = torch.tensor(coords_norm, dtype=torch.float32, device=device).unsqueeze(0)

    demands_np = np.zeros(N, dtype=np.float64)
    demands_np[1:] = problem.demands
    demands = torch.tensor(demands_np, dtype=torch.float32, device=device).unsqueeze(0)

    tw_s = np.zeros(N, dtype=np.float64)
    tw_e = np.full(N, problem.depot_tw_end, dtype=np.float64)
    tw_s[0] = problem.depot_tw_start
    tw_e[0] = problem.depot_tw_end
    tw_s[1:] = problem.tw_start
    tw_e[1:] = problem.tw_end
    tw_start = torch.tensor(tw_s, dtype=torch.float32, device=device).unsqueeze(0)
    tw_end = torch.tensor(tw_e, dtype=torch.float32, device=device).unsqueeze(0)

    svc = np.zeros(N, dtype=np.float64)
    svc[1:] = problem.service_times
    service = torch.tensor(svc, dtype=torch.float32, device=device).unsqueeze(0)

    tt = problem.travel_time_matrices[vehicle_type]
    dk = problem.distance_matrix_meters / 1000.0
    travel_time = torch.tensor(tt, dtype=torch.float32, device=device).unsqueeze(0)
    distance_km = torch.tensor(dk, dtype=torch.float32, device=device).unsqueeze(0)

    dual_np = np.zeros(N, dtype=np.float64)
    dual_np[1:] = dual_values
    duals = torch.tensor(dual_np, dtype=torch.float32, device=device).unsqueeze(0)

    site_np = np.ones(N, dtype=bool)
    site_np[1:] = problem.site_dependency[:, vehicle_idx]
    site_mask = torch.tensor(site_np, dtype=torch.bool, device=device).unsqueeze(0)

    return {
        "coords": coords,
        "demands": demands,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "service_times": service,
        "travel_time": travel_time,
        "distance_km": distance_km,
        "dual_values": duals,
        "site_mask": site_mask,
    }


def _build_arc_tensor(
    arc_set: Optional[Set[Tuple[int, int]]],
    num_nodes: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not arc_set:
        return None
    t = torch.zeros(1, num_nodes, num_nodes, dtype=torch.bool, device=device)
    for i, j in arc_set:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            t[0, i, j] = True
    return t
