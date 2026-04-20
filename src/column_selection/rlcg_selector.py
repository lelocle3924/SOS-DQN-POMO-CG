"""RLCG-style single-column selector built on Graph-DQN."""

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
    select_candidate_indices,
    train_graph_dqn_supervised,
)
from src.master_problem import solve_master_problem
from src.pricing_orchestrator import PricingOrchestrator


class RLCGSelector(AbstractColumnSelector):
    """Select one candidate column per iteration."""

    def __init__(
        self,
        q_model: BipartiteGraphQNetwork,
        device: torch.device,
        rc_tolerance: float = -1e-6,
    ) -> None:
        self.q_model = q_model
        self.device = device
        self.rc_tolerance = rc_tolerance
        self.tracker = ColumnFeatureTracker()

    def select_columns(
        self,
        state: ColumnSelectionState,
        candidate_routes: Sequence[Route],
    ) -> List[Route]:
        if not candidate_routes:
            return []

        self.tracker.update_from_rmp(state.column_pool.routes, state.rmp_result.route_weights)
        graph_state = build_bipartite_graph_state(
            column_pool=state.column_pool,
            rmp_result=state.rmp_result,
            candidate_routes=candidate_routes,
            num_customers=state.problem.num_customers,
            tracker=self.tracker,
        )
        valid_action_mask = [
            bool(route.reduced_cost is not None and route.reduced_cost < self.rc_tolerance)
            for route in candidate_routes
        ]
        selected_local_indices = select_candidate_indices(
            model=self.q_model,
            state=graph_state,
            max_selected=1,
            action_mask=valid_action_mask,
            device=self.device,
        )
        if selected_local_indices:
            selected = candidate_routes[selected_local_indices[0]]
            if selected.reduced_cost is not None and selected.reduced_cost < self.rc_tolerance:
                return [selected]

        fallback = min(
            candidate_routes,
            key=lambda route: route.reduced_cost if route.reduced_cost is not None else 1e18,
        )
        return [fallback] if fallback.reduced_cost is not None and fallback.reduced_cost < self.rc_tolerance else []


def collect_rlcg_training_samples(
    problems: Sequence,
    orchestrator: PricingOrchestrator,
    build_initial_pool_fn,
    config: dict,
    device: torch.device,
    max_iterations_per_problem: int = 4,
) -> List[Tuple]:
    """Build imitation samples: teacher picks most-negative reduced-cost column."""
    samples: List[Tuple] = []
    tracker = ColumnFeatureTracker()

    for problem in problems:
        pool = ColumnPool()
        build_initial_pool_fn(problem, pool)
        bb_constraints = {"forbidden_arcs": set(), "enforced_arcs": set()}

        for iteration_index in range(max_iterations_per_problem):
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

            graph_state = build_bipartite_graph_state(
                column_pool=pool,
                rmp_result=rmp,
                candidate_routes=candidates,
                num_customers=problem.num_customers,
                tracker=tracker,
            )
            best_local_index = int(
                np.argmin(
                    [
                        route.reduced_cost if route.reduced_cost is not None else 1e18
                        for route in candidates
                    ]
                )
            )
            samples.append((graph_state, best_local_index))

            teacher_route = candidates[best_local_index]
            pool.add_route(
                vehicle_type=teacher_route.vehicle_type,
                customer_indices=teacher_route.customer_indices,
                total_cost=teacher_route.total_cost,
                total_distance_km=teacher_route.total_distance_km,
                total_time_hours=teacher_route.total_time_hours,
                reduced_cost=teacher_route.reduced_cost,
            )

    return samples


def train_rlcg_selector(
    q_model: BipartiteGraphQNetwork,
    training_samples: Sequence[Tuple],
    device: torch.device,
    num_epochs: int = 20,
    learning_rate: float = 1.0e-3,
    progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> dict:
    """Train RLCG selector with one-action imitation targets."""
    return train_graph_dqn_supervised(
        model=q_model,
        training_samples=training_samples,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        progress_callback=progress_callback,
    )
