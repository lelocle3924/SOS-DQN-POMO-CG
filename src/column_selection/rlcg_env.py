"""RLCG rollout environment for DQN-style column selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.column_pool import ColumnPool, Route
from src.data_loader import ProblemData
from src.graph_dqn import ColumnFeatureTracker, BipartiteGraphState, build_bipartite_graph_state
from src.master_problem import MasterProblemResult, solve_master_problem


CandidateGenerator = Callable[[ProblemData, np.ndarray, Dict[str, float]], List[Route]]
InitialPoolBuilder = Callable[[ProblemData, ColumnPool], None]


@dataclass
class EnvironmentObservation:
    """Current decision point for one CG iteration."""

    state: Optional[BipartiteGraphState]
    action_mask: List[bool]
    done: bool
    info: Dict[str, float | int | str]


class RLCGEnvironment:
    """Deterministic CG environment matching RLCG transition semantics."""

    def __init__(
        self,
        problem: ProblemData,
        candidate_generator: CandidateGenerator,
        build_initial_pool: InitialPoolBuilder,
        reduced_cost_tolerance: float,
        alpha: float,
        max_episode_steps: int,
    ) -> None:
        self.problem = problem
        self._candidate_generator = candidate_generator
        self._build_initial_pool = build_initial_pool
        self._reduced_cost_tolerance = float(reduced_cost_tolerance)
        self._alpha = float(alpha)
        self._max_episode_steps = int(max_episode_steps)
        self._feature_tracker = ColumnFeatureTracker()
        self._column_pool: Optional[ColumnPool] = None
        self._current_rmp: Optional[MasterProblemResult] = None
        self._current_candidates: List[Route] = []
        self._current_state: Optional[BipartiteGraphState] = None
        self._current_action_mask: List[bool] = []
        self._iteration_index = 0
        self._objective_initial = 1.0
        self._done = False
        self._done_reason = ""

    def reset(self) -> EnvironmentObservation:
        self._column_pool = ColumnPool()
        self._build_initial_pool(self.problem, self._column_pool)
        self._feature_tracker = ColumnFeatureTracker()
        self._current_rmp = None
        self._current_candidates = []
        self._current_state = None
        self._current_action_mask = []
        self._iteration_index = 0
        self._done = False
        self._done_reason = ""

        observation = self._refresh_observation()
        if self._current_rmp is not None:
            initial_objective = float(self._current_rmp.objective_value)
            if not np.isfinite(initial_objective):
                initial_objective = 1.0
            self._objective_initial = float(max(abs(initial_objective), 1.0e-6))
        return observation

    def step(self, action_index: int) -> Tuple[EnvironmentObservation, float]:
        if self._done:
            raise RuntimeError("Cannot step an environment that is already done.")
        if self._column_pool is None or self._current_rmp is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if action_index < 0 or action_index >= len(self._current_candidates):
            raise IndexError("Action index out of range for current candidates.")
        if not self._current_action_mask[action_index]:
            raise ValueError("Selected action does not satisfy current action mask.")

        selected_route = self._current_candidates[action_index]
        self._column_pool.add_route(
            vehicle_type=selected_route.vehicle_type,
            customer_indices=selected_route.customer_indices,
            total_cost=selected_route.total_cost,
            total_distance_km=selected_route.total_distance_km,
            total_time_hours=selected_route.total_time_hours,
            reduced_cost=selected_route.reduced_cost,
        )
        previous_objective = float(self._current_rmp.objective_value)
        self._iteration_index += 1

        observation = self._refresh_observation()
        next_objective = float(observation.info.get("objective_value", previous_objective))
        scaled_improvement = self._alpha * (previous_objective - next_objective) / self._objective_initial
        reward = float(scaled_improvement - 1.0)
        return observation, reward

    def _refresh_observation(self) -> EnvironmentObservation:
        if self._column_pool is None:
            raise RuntimeError("Environment must be reset before observation is available.")
        if self._iteration_index >= self._max_episode_steps:
            self._done = True
            self._done_reason = "max_episode_steps"
            return EnvironmentObservation(
                state=None,
                action_mask=[],
                done=True,
                info={
                    "done_reason": self._done_reason,
                    "iteration": self._iteration_index,
                },
            )

        rmp_result = solve_master_problem(self._column_pool, self.problem)
        if rmp_result.status != "OPTIMAL":
            self._done = True
            self._done_reason = "rmp_not_optimal"
            return EnvironmentObservation(
                state=None,
                action_mask=[],
                done=True,
                info={
                    "done_reason": self._done_reason,
                    "iteration": self._iteration_index,
                    "rmp_status": rmp_result.status,
                },
            )

        self._feature_tracker.update_from_rmp(self._column_pool.routes, rmp_result.route_weights)
        candidates = self._candidate_generator(
            self.problem,
            rmp_result.dual_values,
            rmp_result.vehicle_dual_values,
        )
        action_mask = [
            bool(route.reduced_cost is not None and route.reduced_cost < self._reduced_cost_tolerance)
            for route in candidates
        ]
        if not candidates or not any(action_mask):
            self._done = True
            self._done_reason = "cg_converged"
            return EnvironmentObservation(
                state=None,
                action_mask=[],
                done=True,
                info={
                    "done_reason": self._done_reason,
                    "iteration": self._iteration_index,
                    "objective_value": float(rmp_result.objective_value),
                    "candidate_count": int(len(candidates)),
                },
            )

        graph_state = build_bipartite_graph_state(
            column_pool=self._column_pool,
            rmp_result=rmp_result,
            candidate_routes=candidates,
            num_customers=self.problem.num_customers,
            tracker=self._feature_tracker,
        )
        self._current_rmp = rmp_result
        self._current_candidates = candidates
        self._current_state = graph_state
        self._current_action_mask = action_mask
        return EnvironmentObservation(
            state=graph_state,
            action_mask=action_mask,
            done=False,
            info={
                "done_reason": "",
                "iteration": self._iteration_index,
                "objective_value": float(rmp_result.objective_value),
                "candidate_count": int(len(candidates)),
                "valid_action_count": int(sum(1 for is_valid in action_mask if is_valid)),
            },
        )
