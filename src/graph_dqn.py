"""Graph-DQN components for column selection in column generation.

This module implements a bipartite graph state and a DQN-like Q-network
that scores candidate action columns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.column_pool import ColumnPool, Route
from src.master_problem import MasterProblemResult


RouteKey = Tuple[str, Tuple[int, ...]]
logger = logging.getLogger(__name__)


def make_route_key(route: Route) -> RouteKey:
    return route.vehicle_type, tuple(sorted(route.customer_indices))


class ColumnFeatureTracker:
    """Track basis dynamics needed for column node features (e-h)."""

    def __init__(self) -> None:
        self._history: Dict[RouteKey, Dict[str, float]] = {}

    def update_from_rmp(self, routes: Sequence[Route], route_weights: np.ndarray) -> None:
        route_count = min(len(routes), len(route_weights))
        seen_now = set()

        for index in range(route_count):
            route = routes[index]
            key = make_route_key(route)
            seen_now.add(key)
            in_basis = float(route_weights[index]) > 1e-8

            record = self._history.get(
                key,
                {
                    "in_basis_count": 0.0,
                    "out_basis_count": 0.0,
                    "left_last_iter": 0.0,
                    "entered_last_iter": 0.0,
                    "prev_in_basis": 0.0,
                },
            )
            prev_in_basis = bool(record["prev_in_basis"] > 0.5)
            entered = float((not prev_in_basis) and in_basis)
            left = float(prev_in_basis and (not in_basis))

            if in_basis:
                record["in_basis_count"] += 1.0
                record["out_basis_count"] = 0.0
            else:
                record["out_basis_count"] += 1.0
                record["in_basis_count"] = 0.0

            record["entered_last_iter"] = entered
            record["left_last_iter"] = left
            record["prev_in_basis"] = 1.0 if in_basis else 0.0
            self._history[key] = record

        # For routes not in the current RMP vector, clear one-step flags.
        for key, record in self._history.items():
            if key not in seen_now:
                record["entered_last_iter"] = 0.0
                record["left_last_iter"] = 0.0

    def get_dynamic_features(self, route_key: RouteKey) -> Tuple[float, float, float, float]:
        record = self._history.get(route_key)
        if record is None:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(record["in_basis_count"]),
            float(record["out_basis_count"]),
            float(record["left_last_iter"]),
            float(record["entered_last_iter"]),
        )


@dataclass
class BipartiteGraphState:
    """State representation for Graph-DQN."""

    column_features: torch.Tensor
    constraint_features: torch.Tensor
    edge_column_index: torch.Tensor
    edge_constraint_index: torch.Tensor
    action_node_indices: torch.Tensor

    def to(self, device: torch.device) -> "BipartiteGraphState":
        return BipartiteGraphState(
            column_features=self.column_features.to(device),
            constraint_features=self.constraint_features.to(device),
            edge_column_index=self.edge_column_index.to(device),
            edge_constraint_index=self.edge_constraint_index.to(device),
            action_node_indices=self.action_node_indices.to(device),
        )


def clone_graph_state_to_cpu(state: BipartiteGraphState) -> BipartiteGraphState:
    """Create a detached CPU copy for replay storage."""
    return BipartiteGraphState(
        column_features=state.column_features.detach().cpu().clone(),
        constraint_features=state.constraint_features.detach().cpu().clone(),
        edge_column_index=state.edge_column_index.detach().cpu().clone(),
        edge_constraint_index=state.edge_constraint_index.detach().cpu().clone(),
        action_node_indices=state.action_node_indices.detach().cpu().clone(),
    )


@dataclass
class ReplayTransition:
    """Single off-policy transition for DQN updates."""

    state: BipartiteGraphState
    action_local_index: int
    reward: float
    next_state: Optional[BipartiteGraphState]
    done: bool
    next_action_mask: Optional[List[bool]]


class ReplayBuffer:
    """Simple bounded replay buffer with uniform sampling."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self._capacity = int(capacity)
        self._buffer: Deque[ReplayTransition] = deque(maxlen=self._capacity)

    def add(self, transition: ReplayTransition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[ReplayTransition]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        sample_size = min(batch_size, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=sample_size, replace=False)
        return [self._buffer[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._buffer)


def _compute_route_reduced_cost(
    route: Route,
    dual_values: np.ndarray,
    vehicle_dual_values: Dict[str, float],
) -> float:
    if route.reduced_cost is not None:
        return float(route.reduced_cost)
    dual_sum = sum(float(dual_values[c]) for c in route.customer_indices)
    vehicle_dual = float(vehicle_dual_values.get(route.vehicle_type, 0.0))
    return float(route.total_cost - dual_sum - vehicle_dual)


def build_bipartite_graph_state(
    column_pool: ColumnPool,
    rmp_result: MasterProblemResult,
    candidate_routes: Sequence[Route],
    num_customers: int,
    tracker: ColumnFeatureTracker,
) -> BipartiteGraphState:
    """Build graph state with required 9 column-node features.

    Column features in order:
      (a) reduced_cost
      (b) connectivity
      (c) solution_value
      (d) total_route_cost
      (e) in_basis_count
      (f) out_basis_count
      (g) left_last_iteration
      (h) entered_last_iteration
      (i) action_node_indicator
    """
    existing_routes = list(column_pool.routes)
    total_column_count = len(existing_routes) + len(candidate_routes)

    column_features_np = np.zeros((total_column_count, 9), dtype=np.float32)
    edge_columns: List[int] = []
    edge_constraints: List[int] = []

    # Existing columns
    for col_idx, route in enumerate(existing_routes):
        route_key = make_route_key(route)
        reduced_cost = _compute_route_reduced_cost(
            route, rmp_result.dual_values, rmp_result.vehicle_dual_values
        )
        connectivity = float(len(set(route.customer_indices)))
        solution_value = (
            float(rmp_result.route_weights[col_idx])
            if col_idx < len(rmp_result.route_weights)
            else 0.0
        )
        in_count, out_count, left_last, entered_last = tracker.get_dynamic_features(route_key)

        column_features_np[col_idx, :] = np.array(
            [
                reduced_cost,
                connectivity,
                solution_value,
                float(route.total_cost),
                in_count,
                out_count,
                left_last,
                entered_last,
                0.0,  # action flag
            ],
            dtype=np.float32,
        )

        for customer_idx in route.customer_indices:
            if 0 <= customer_idx < num_customers:
                edge_columns.append(col_idx)
                edge_constraints.append(customer_idx)

    # Candidate action columns
    action_node_indices: List[int] = []
    for cand_offset, route in enumerate(candidate_routes):
        col_idx = len(existing_routes) + cand_offset
        action_node_indices.append(col_idx)

        route_key = make_route_key(route)
        reduced_cost = _compute_route_reduced_cost(
            route, rmp_result.dual_values, rmp_result.vehicle_dual_values
        )
        connectivity = float(len(set(route.customer_indices)))
        in_count, out_count, left_last, entered_last = tracker.get_dynamic_features(route_key)

        column_features_np[col_idx, :] = np.array(
            [
                reduced_cost,
                connectivity,
                0.0,  # candidate solution value
                float(route.total_cost),
                in_count,
                out_count,
                left_last,
                entered_last,
                1.0,  # action flag
            ],
            dtype=np.float32,
        )

        for customer_idx in route.customer_indices:
            if 0 <= customer_idx < num_customers:
                edge_columns.append(col_idx)
                edge_constraints.append(customer_idx)

    # Constraint features: [dual_value, connectivity]
    constraint_connectivity = np.zeros(num_customers, dtype=np.float32)
    for customer_idx in edge_constraints:
        constraint_connectivity[customer_idx] += 1.0

    dual_values = rmp_result.dual_values
    if len(dual_values) < num_customers:
        dual_pad = np.zeros(num_customers, dtype=np.float32)
        dual_pad[: len(dual_values)] = dual_values.astype(np.float32)
        dual_values_np = dual_pad
    else:
        dual_values_np = dual_values[:num_customers].astype(np.float32)

    constraint_features_np = np.stack(
        [dual_values_np, constraint_connectivity], axis=1
    ).astype(np.float32)

    if edge_columns:
        edge_column_tensor = torch.tensor(edge_columns, dtype=torch.long)
        edge_constraint_tensor = torch.tensor(edge_constraints, dtype=torch.long)
    else:
        edge_column_tensor = torch.zeros(0, dtype=torch.long)
        edge_constraint_tensor = torch.zeros(0, dtype=torch.long)

    return BipartiteGraphState(
        column_features=torch.tensor(column_features_np, dtype=torch.float32),
        constraint_features=torch.tensor(constraint_features_np, dtype=torch.float32),
        edge_column_index=edge_column_tensor,
        edge_constraint_index=edge_constraint_tensor,
        action_node_indices=torch.tensor(action_node_indices, dtype=torch.long),
    )


class BipartiteGraphQNetwork(nn.Module):
    """Two-phase bipartite message passing network for action Q-values."""

    def __init__(
        self,
        column_feature_dim: int = 9,
        constraint_feature_dim: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.column_encoder = nn.Linear(column_feature_dim, hidden_dim)
        self.constraint_encoder = nn.Linear(constraint_feature_dim, hidden_dim)
        self.column_to_constraint = nn.Linear(hidden_dim, hidden_dim)
        self.constraint_to_column = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: BipartiteGraphState) -> torch.Tensor:
        col_hidden = F.relu(self.column_encoder(state.column_features))
        con_hidden = F.relu(self.constraint_encoder(state.constraint_features))

        edge_col = state.edge_column_index
        edge_con = state.edge_constraint_index

        if edge_col.numel() > 0:
            # Phase 1: columns -> constraints
            msg_col = F.relu(self.column_to_constraint(col_hidden))
            agg_con = torch.zeros_like(con_hidden)
            agg_con.index_add_(0, edge_con, msg_col[edge_col])
            con_degree = torch.bincount(
                edge_con, minlength=con_hidden.size(0)
            ).float().unsqueeze(1).clamp(min=1.0)
            con_hidden = F.relu(con_hidden + agg_con / con_degree)

            # Phase 2: constraints -> columns
            msg_con = F.relu(self.constraint_to_column(con_hidden))
            agg_col = torch.zeros_like(col_hidden)
            agg_col.index_add_(0, edge_col, msg_con[edge_con])
            col_degree = torch.bincount(
                edge_col, minlength=col_hidden.size(0)
            ).float().unsqueeze(1).clamp(min=1.0)
            col_hidden = F.relu(col_hidden + agg_col / col_degree)

        return self.q_head(col_hidden).squeeze(-1)


def train_graph_dqn_supervised(
    model: BipartiteGraphQNetwork,
    training_samples: Sequence[Tuple[BipartiteGraphState, int]],
    num_epochs: int = 20,
    learning_rate: float = 1.0e-3,
    device: torch.device | None = None,
    progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    """Train on graph states with teacher action labels (best candidate index)."""
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses: List[float] = []
    valid_samples = [sample for sample in training_samples if sample[0].action_node_indices.numel() > 0]
    if not valid_samples:
        return {"loss": 0.0, "num_samples": 0}

    train_start_time = time.time()
    for epoch_index in range(num_epochs):
        permutation = np.random.permutation(len(valid_samples))
        total_loss = 0.0
        count = 0
        for sample_idx in permutation:
            state, target_local_action = valid_samples[sample_idx]
            if target_local_action < 0:
                continue
            state_device = state.to(device)
            q_all = model(state_device)
            action_q = q_all[state_device.action_node_indices]
            if action_q.numel() == 0:
                continue
            if target_local_action >= action_q.numel():
                continue
            target_tensor = torch.tensor([target_local_action], dtype=torch.long, device=device)
            loss = F.cross_entropy(action_q.unsqueeze(0), target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
        epoch_losses.append(total_loss / max(1, count))
        completed_epochs = epoch_index + 1
        progress = completed_epochs / max(1, num_epochs)
        elapsed_seconds = time.time() - train_start_time
        eta_seconds = (elapsed_seconds / progress) - elapsed_seconds if progress > 0 else float("nan")
        logger.info(
            "Graph-DQN train epoch %d/%d (%.1f%%)  loss=%.6f  eta=%.1fs",
            completed_epochs,
            num_epochs,
            progress * 100.0,
            epoch_losses[-1],
            eta_seconds,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": float(completed_epochs),
                    "num_epochs": float(num_epochs),
                    "progress_percent": float(progress * 100.0),
                    "loss": float(epoch_losses[-1]),
                    "eta_seconds": float(eta_seconds),
                    "elapsed_seconds": float(elapsed_seconds),
                }
            )

    return {"loss": float(np.mean(epoch_losses)), "num_samples": len(valid_samples)}


def select_candidate_indices(
    model: BipartiteGraphQNetwork,
    state: BipartiteGraphState,
    max_selected: int = 1,
    action_mask: Optional[Sequence[bool]] = None,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    device: torch.device | None = None,
) -> List[int]:
    """Return selected candidate indices (local to candidate list)."""
    if device is None:
        device = torch.device("cpu")
    if state.action_node_indices.numel() == 0:
        return []

    model.eval()
    with torch.no_grad():
        state_device = state.to(device)
        q_all = model(state_device)
        action_q = q_all[state_device.action_node_indices]
        num_actions = int(action_q.numel())
        if num_actions == 0:
            return []

        if action_mask is None:
            valid_action_indices = torch.arange(num_actions, device=action_q.device, dtype=torch.long)
        else:
            if len(action_mask) != num_actions:
                raise ValueError("Action mask length must match candidate action count.")
            valid_action_indices = torch.tensor(action_mask, device=action_q.device, dtype=torch.bool).nonzero(
                as_tuple=False
            ).view(-1)
        if valid_action_indices.numel() == 0:
            return []

        if rng is None:
            rng = np.random.default_rng()
        if max_selected == 1 and epsilon > 0.0 and float(rng.random()) < epsilon:
            random_choice = int(valid_action_indices[int(rng.integers(valid_action_indices.numel()))].item())
            return [random_choice]

        masked_q = torch.full_like(action_q, fill_value=-float("inf"))
        masked_q[valid_action_indices] = action_q[valid_action_indices]
        top_k = int(min(max_selected, valid_action_indices.numel()))
        if top_k <= 0:
            return []
        _, indices = torch.topk(masked_q, k=top_k, largest=True)
        return [int(i.item()) for i in indices]
