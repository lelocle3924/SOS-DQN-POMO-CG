"""FFCG-style variable-size family selector with sequential STOP logic."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.column_pool import ColumnPool, Route
from src.column_selection.base import AbstractColumnSelector, ColumnSelectionState
from src.graph_dqn import (
    BipartiteGraphQNetwork,
    ColumnFeatureTracker,
    build_bipartite_graph_state,
    train_graph_dqn_supervised,
)
from src.master_problem import solve_master_problem
from src.pricing_orchestrator import PricingOrchestrator


class FFCGSelector(AbstractColumnSelector):
    """Select a variable-size subset by sequential marginal picks."""

    def __init__(
        self,
        q_model: BipartiteGraphQNetwork,
        device: torch.device,
        rc_tolerance: float = -1e-6,
        max_family_size: int = 5,
        stop_q_threshold: float = 0.0,
    ) -> None:
        self.q_model = q_model
        self.device = device
        self.rc_tolerance = rc_tolerance
        self.max_family_size = max_family_size
        self.stop_q_threshold = stop_q_threshold
        self.tracker = ColumnFeatureTracker()

    def select_columns(
        self,
        state: ColumnSelectionState,
        candidate_routes: Sequence[Route],
    ) -> List[Route]:
        if not candidate_routes:
            return []

        self.tracker.update_from_rmp(state.column_pool.routes, state.rmp_result.route_weights)
        remaining_routes = list(candidate_routes)
        selected_routes: List[Route] = []

        while remaining_routes and len(selected_routes) < self.max_family_size:
            graph_state = build_bipartite_graph_state(
                column_pool=state.column_pool,
                rmp_result=state.rmp_result,
                candidate_routes=remaining_routes,
                num_customers=state.problem.num_customers,
                tracker=self.tracker,
            ).to(self.device)

            with torch.no_grad():
                q_all = self.q_model(graph_state)
                action_q = q_all[graph_state.action_node_indices]
                if action_q.numel() == 0:
                    break
                best_local_index = int(torch.argmax(action_q).item())
                best_q_value = float(action_q[best_local_index].item())

            # Sequential STOP decision.
            if best_q_value <= self.stop_q_threshold:
                break

            selected = remaining_routes.pop(best_local_index)
            if selected.reduced_cost is not None and selected.reduced_cost < self.rc_tolerance:
                selected_routes.append(selected)

        if not selected_routes and candidate_routes:
            fallback = min(
                candidate_routes,
                key=lambda route: route.reduced_cost if route.reduced_cost is not None else 1e18,
            )
            if fallback.reduced_cost is not None and fallback.reduced_cost < self.rc_tolerance:
                selected_routes = [fallback]
        return selected_routes


def collect_ffcg_training_samples(
    problems: Sequence,
    orchestrator: PricingOrchestrator,
    build_initial_pool_fn,
    config: dict,
    device: torch.device,
    max_iterations_per_problem: int = 4,
    max_family_size: int = 5,
) -> List[Tuple]:
    """Build sequential teacher samples for FFCG.

    Teacher policy: repeatedly choose the best remaining reduced-cost column.
    """
    samples: List[Tuple] = []
    tracker = ColumnFeatureTracker()

    for problem in problems:
        pool = ColumnPool()
        build_initial_pool_fn(problem, pool)
        bb_constraints = {"forbidden_arcs": set(), "enforced_arcs": set()}

        for _ in range(max_iterations_per_problem):
            rmp = solve_master_problem(pool, problem)
            if rmp.status != "OPTIMAL":
                break
            tracker.update_from_rmp(pool.routes, rmp.route_weights)

            candidates = orchestrator.generate_columns(
                problem,
                rmp.dual_values,
                rmp.vehicle_dual_values,
                bb_constraints,
            )
            if not candidates:
                break

            remaining = list(candidates)
            selected_count = 0
            while remaining and selected_count < max_family_size:
                graph_state = build_bipartite_graph_state(
                    column_pool=pool,
                    rmp_result=rmp,
                    candidate_routes=remaining,
                    num_customers=problem.num_customers,
                    tracker=tracker,
                )
                teacher_local_index = int(
                    np.argmin(
                        [
                            route.reduced_cost if route.reduced_cost is not None else 1e18
                            for route in remaining
                        ]
                    )
                )
                if remaining[teacher_local_index].reduced_cost is None:
                    break
                if remaining[teacher_local_index].reduced_cost >= config["column_generation"].get(
                    "reduced_cost_tolerance", -1e-6
                ):
                    break

                samples.append((graph_state, teacher_local_index))
                teacher_route = remaining.pop(teacher_local_index)
                pool.add_route(
                    vehicle_type=teacher_route.vehicle_type,
                    customer_indices=teacher_route.customer_indices,
                    total_cost=teacher_route.total_cost,
                    total_distance_km=teacher_route.total_distance_km,
                    total_time_hours=teacher_route.total_time_hours,
                    reduced_cost=teacher_route.reduced_cost,
                )
                selected_count += 1

    return samples


def train_ffcg_selector(
    q_model: BipartiteGraphQNetwork,
    training_samples: Sequence[Tuple],
    device: torch.device,
    num_epochs: int = 20,
    learning_rate: float = 1.0e-3,
    progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> dict:
    """Train FFCG selector with sequential action targets."""
    return train_graph_dqn_supervised(
        model=q_model,
        training_samples=training_samples,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        progress_callback=progress_callback,
    )
