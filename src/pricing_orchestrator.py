"""Pricing Orchestrator: 3-strike POMO with DP Label Setting fallback.

Flow per CG iteration:
    1. POMO greedy rollout  → return if negative RC columns found
    2. POMO sampling (temp 1.5) → return if found
    3. POMO sampling (temp 2.0) → return if found
    4. DP Label Setting fallback

If the DP fallback fires on >30 % of calls, a diagnostics report is written.
"""

import gc
import logging
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.column_pool import Route
from src.data_loader import ProblemData
from src.pomo_env import VRPTWEnvironment
from src.pomo_model import POMOModel
from src.pomo_trainer import build_node_features
from src.pricing_problem import _build_arc_tensor, _build_instance_tensors

logger = logging.getLogger(__name__)


class PricingOrchestrator:
    """Orchestrates POMO rollouts + DP fallback to generate negative RC columns."""

    def __init__(self, model: POMOModel, device: torch.device, config: Dict):
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()

        self.total_pricing_calls: int = 0
        self.dp_fallback_count: int = 0
        self.pomo_success_count: int = 0
        self._diagnostics_written: bool = False

    # ------------------------------------------------------------------
    # Public entry-point (called once per CG iteration per B&B node)
    # ------------------------------------------------------------------

    def generate_columns(
        self,
        problem: ProblemData,
        dual_values: np.ndarray,
        vehicle_dual_values: Dict[str, float],
        bb_constraints: Dict[str, Set[Tuple[int, int]]],
    ) -> List[Route]:
        self.total_pricing_calls += 1
        forbidden_arcs = bb_constraints.get("forbidden_arcs", set())
        enforced_arcs = bb_constraints.get("enforced_arcs", set())
        rc_tol = self.config["column_generation"].get(
            "reduced_cost_tolerance", -1e-5,
        )

        # Strike 1 — greedy
        routes = self._try_pomo(
            problem, dual_values, vehicle_dual_values,
            forbidden_arcs, enforced_arcs, rc_tol, "greedy", 1.0,
        )
        if routes:
            self.pomo_success_count += 1
            return routes

        # Strike 2 — sampling temperature 1.5
        routes = self._try_pomo(
            problem, dual_values, vehicle_dual_values,
            forbidden_arcs, enforced_arcs, rc_tol, "sampling", 1.5,
        )
        if routes:
            self.pomo_success_count += 1
            return routes

        # Strike 3 — sampling temperature 2.0
        routes = self._try_pomo(
            problem, dual_values, vehicle_dual_values,
            forbidden_arcs, enforced_arcs, rc_tol, "sampling", 2.0,
        )
        if routes:
            self.pomo_success_count += 1
            return routes

        # DP fallback
        self.dp_fallback_count += 1
        logger.warning(
            "POMO failed 3 strikes → DP fallback (call %d / %d).",
            self.dp_fallback_count, self.total_pricing_calls,
        )
        routes = self._run_dp_fallback(
            problem, dual_values, vehicle_dual_values,
            forbidden_arcs, enforced_arcs, rc_tol,
        )

        dp_ratio = self.dp_fallback_count / max(1, self.total_pricing_calls)
        if dp_ratio > 0.3 and not self._diagnostics_written:
            self._write_pomo_diagnostics(dp_ratio)

        return routes

    # ------------------------------------------------------------------
    # POMO attempt (all vehicle types)
    # ------------------------------------------------------------------

    def _try_pomo(
        self,
        problem: ProblemData,
        dual_values: np.ndarray,
        vehicle_dual_values: Dict[str, float],
        forbidden_arcs: Set[Tuple[int, int]],
        enforced_arcs: Set[Tuple[int, int]],
        rc_tol: float,
        decode_method: str,
        temperature: float,
    ) -> List[Route]:
        new_routes: List[Route] = []
        for v_idx, vtype in enumerate(problem.vehicle_types):
            accessible = [
                c for c in range(problem.num_customers)
                if problem.site_dependency[c, v_idx]
            ]
            if not accessible:
                continue
            generated = self._run_rollout(
                problem, dual_values,
                vehicle_dual_values.get(vtype, 0.0),
                vtype, v_idx, accessible,
                forbidden_arcs, enforced_arcs,
                decode_method, temperature,
            )
            for cust_idx, cost, dist, time_h, rc in generated:
                if rc < rc_tol:
                    new_routes.append(Route(
                        route_id=-1,
                        vehicle_type=vtype,
                        customer_indices=cust_idx,
                        visit_sequence=cust_idx,
                        total_cost=cost,
                        total_distance_km=dist,
                        total_time_hours=time_h,
                        reduced_cost=rc,
                    ))
        return new_routes

    # ------------------------------------------------------------------
    # Single vehicle-type POMO rollout
    # ------------------------------------------------------------------

    def _run_rollout(
        self,
        problem: ProblemData,
        dual_values: np.ndarray,
        vehicle_dual: float,
        vehicle_type: str,
        vehicle_idx: int,
        accessible_customers: List[int],
        forbidden_arcs: Set[Tuple[int, int]],
        enforced_arcs: Set[Tuple[int, int]],
        decode_method: str,
        temperature: float,
    ) -> List[Tuple[List[int], float, float, float, float]]:
        num_nodes = problem.num_customers + 1
        capacity = problem.vehicle_capacity[vehicle_type]
        fixed_cost = problem.vehicle_fixed_cost[vehicle_type]
        cost_per_km = problem.vehicle_cost_per_km[vehicle_type]
        cost_per_hour = problem.vehicle_cost_per_hour[vehicle_type]
        depot_tw_end = problem.depot_tw_end

        P = len(accessible_customers)
        if P == 0:
            return []
        B = P

        instance = _build_instance_tensors(
            problem, dual_values, vehicle_type, vehicle_idx, self.device,
        )

        forbidden_t = _build_arc_tensor(forbidden_arcs, num_nodes, self.device)
        enforced_t = _build_arc_tensor(enforced_arcs, num_nodes, self.device)

        def rep(t: torch.Tensor) -> torch.Tensor:
            return t.expand(B, *[-1] * (t.dim() - 1)).contiguous()

        node_feat = build_node_features(instance, capacity, depot_tw_end)
        self.model.set_decode_context(capacity, depot_tw_end)
        env = VRPTWEnvironment(self.device)

        with torch.no_grad():
            enc_out, graph_emb = self.model.encode(rep(node_feat))

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
                forbidden_arcs=(
                    rep(forbidden_t) if forbidden_t is not None else None
                ),
                enforced_arcs=(
                    rep(enforced_t) if enforced_t is not None else None
                ),
            )

            # Force distinct first customers for POMO diversity
            pomo_starts = torch.tensor(
                [c + 1 for c in accessible_customers],
                dtype=torch.long, device=self.device,
            )

            mask = env.get_action_mask(state)
            forced_mask = torch.ones(
                B, num_nodes, dtype=torch.bool, device=self.device,
            )
            forced_mask[torch.arange(B, device=self.device), pomo_starts] = False
            forced_bad = mask[torch.arange(B, device=self.device), pomo_starts]
            active_mask = torch.where(
                forced_bad.unsqueeze(1), mask, forced_mask,
            )

            logits = self.model.decoder(
                enc_out, graph_emb,
                state.first_customer, state.current_node,
                (state.remaining_capacity / (capacity + 1e-8)).clamp(0, 1),
                (state.current_time / (depot_tw_end + 1e-8)).clamp(0, 1),
                active_mask,
            )

            if decode_method == "greedy":
                action = torch.where(
                    forced_bad, logits.argmax(dim=-1), pomo_starts,
                )
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                action = torch.where(forced_bad, sampled, pomo_starts)

            zero_lp = torch.zeros(
                B, dtype=torch.float32, device=self.device,
            )
            state = env.step(state, action, zero_lp)

            for _ in range(num_nodes + 10):
                if env.is_done(state):
                    break
                step_mask = env.get_action_mask(state)
                action, lp = self.model.decode_step(
                    enc_out, graph_emb, state, step_mask,
                    decode_method=decode_method,
                    temperature=temperature,
                )
                state = env.step(state, action, lp)

            if not env.is_done(state):
                state = env.step(
                    state,
                    torch.zeros(B, dtype=torch.long, device=self.device),
                    torch.zeros(B, dtype=torch.float32, device=self.device),
                )

        # --- Exact post-hoc cost recalculation (float64) ---
        seqs = torch.stack(state.sequences, dim=1).cpu().numpy()
        tt_mat = problem.travel_time_matrices[vehicle_type]
        dk_mat = problem.distance_matrix_meters / 1000.0

        results: List[Tuple[List[int], float, float, float, float]] = []
        for p in range(P):
            cust_indices = [int(n) - 1 for n in seqs[p] if n > 0]
            if not cust_indices:
                continue

            dist_km = 0.0
            cur_node = 0
            cur_time = problem.depot_tw_start

            for c in cust_indices:
                node = c + 1
                dist_km += dk_mat[cur_node, node]
                arrival = cur_time + tt_mat[cur_node, node]
                start_svc = max(arrival, problem.tw_start[c])
                cur_time = start_svc + problem.service_times[c]
                cur_node = node

            dist_km += dk_mat[cur_node, 0]
            return_time = cur_time + tt_mat[cur_node, 0]
            time_h = return_time - problem.depot_tw_start

            route_cost = (
                fixed_cost + cost_per_km * dist_km + cost_per_hour * time_h
            )
            dual_sum = sum(dual_values[c] for c in cust_indices)
            reduced_cost = route_cost - dual_sum - vehicle_dual

            results.append(
                (cust_indices, route_cost, dist_km, time_h, reduced_cost)
            )

        return results

    # ------------------------------------------------------------------
    # DP Label Setting fallback
    # ------------------------------------------------------------------

    def _run_dp_fallback(
        self,
        problem: ProblemData,
        dual_values: np.ndarray,
        vehicle_dual_values: Dict[str, float],
        forbidden_arcs: Set[Tuple[int, int]],
        enforced_arcs: Set[Tuple[int, int]],
        rc_tol: float,
    ) -> List[Route]:
        from src.dp_solver import solve_espprc

        dp_cfg = self.config.get("dp_solver", {})
        beam_width = dp_cfg.get("beam_width", 50)
        max_labels = dp_cfg.get("max_total_labels", 10000)

        new_routes: List[Route] = []
        for v_idx, vtype in enumerate(problem.vehicle_types):
            accessible_count = sum(
                1 for c in range(problem.num_customers)
                if problem.site_dependency[c, v_idx]
            )
            if accessible_count == 0:
                continue

            vehicle_dual = vehicle_dual_values.get(vtype, 0.0)
            results = solve_espprc(
                problem, dual_values, vtype, v_idx,
                forbidden_arcs, enforced_arcs,
                vehicle_dual=vehicle_dual,
                beam_width=beam_width,
                max_total_labels=max_labels,
            )
            for cust_idx, cost, dist, time_h, rc in results:
                if rc < rc_tol:
                    new_routes.append(Route(
                        route_id=-1,
                        vehicle_type=vtype,
                        customer_indices=cust_idx,
                        visit_sequence=cust_idx,
                        total_cost=cost,
                        total_distance_km=dist,
                        total_time_hours=time_h,
                        reduced_cost=rc,
                    ))

        gc.collect()
        return new_routes

    # ------------------------------------------------------------------
    # POMO failure diagnostics
    # ------------------------------------------------------------------

    def _write_pomo_diagnostics(self, dp_ratio: float) -> None:
        self._diagnostics_written = True
        report_path = os.path.join(
            self.config.get("logging", {}).get("results_dir", "results"),
            "POMO_REWARD_DIAGNOSTICS.md",
        )

        total = self.total_pricing_calls
        dp = self.dp_fallback_count
        pomo_ok = self.pomo_success_count

        content = f"""# POMO Reward Diagnostics

**Generated automatically** because the DP fallback ratio exceeded 30 %.

## Statistics
| Metric | Value |
|--------|-------|
| Total pricing calls | {total} |
| POMO successes | {pomo_ok} |
| DP fallback calls | {dp} |
| DP fallback ratio | {dp_ratio:.1%} |

## Failure Mechanism Analysis

The high DP fallback ratio indicates that POMO is consistently failing to
find negative reduced cost columns within 3 sampling attempts.  Possible
root causes:

1. **Reward–cost misalignment during training.**
   If the REINFORCE reward used during pre-training does not match the
   exact cost formula `FixedCost + CostPerKm * dist + CostPerHour * time`,
   the learned policy optimises for the wrong objective.  The model may
   confidently produce routes that are sub-optimal under the real metric.

2. **Lack of explicit edge-feature attention.**
   The current Attention Model processes node features only.  Real-world
   edge costs (from the OSRM matrix) are encoded *implicitly* through the
   REINFORCE reward signal.  When the topology is highly non-Euclidean
   (e.g. one-way streets, highways) the model cannot distinguish cheap
   from expensive arcs without an edge-aware encoder layer.

3. **Covariate shift in dual values.**
   During training, dual values are synthetic and uniformly distributed.
   During CG inference, duals evolve with the LP relaxation and may
   concentrate in narrow ranges or spike for hard-to-cover customers.
   Normalisation (max-scaling to [0, 1]) mitigates this but does not
   eliminate it entirely.

## Proposed Improvements

- **Short term**: retrain POMO on sub-graphs sampled from the real
  distance matrix (Phase 0 fix) so the reward reflects true edge costs.
- **Medium term**: add a Graph Attention Network (GAT) encoder layer
  that takes the distance / travel-time matrix as edge features.
  This gives the model direct access to arc costs at every attention
  step, removing the need to learn them purely from reward.
- **Long term**: augment training with an auxiliary loss that predicts
  the reduced cost of each rollout (supervised signal from the DP
  solver), creating a hybrid RL + supervised approach.
"""
        try:
            os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            logger.warning("Wrote POMO diagnostics to %s", report_path)
        except OSError as exc:
            logger.error("Could not write diagnostics: %s", exc)
