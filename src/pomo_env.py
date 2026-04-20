"""Batched VRPTW environment for POMO with site-dependency and branching masks.

Node convention:  node 0 = depot,  node c+1 = customer c  (0-based c).
"""

import torch
from typing import List, Optional


class VRPTWState:
    """Mutable, batched state carried through a POMO rollout."""

    def __init__(self, batch_size: int, num_nodes: int,
                 device: torch.device) -> None:
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.device = device

        self.current_node = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self.visited = torch.zeros(
            batch_size, num_nodes, dtype=torch.bool, device=device
        )
        self.remaining_capacity = torch.zeros(
            batch_size, dtype=torch.float32, device=device
        )
        self.current_time = torch.zeros(
            batch_size, dtype=torch.float32, device=device
        )
        self.first_customer = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self.route_started = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        self.done = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

        self.log_probs: List[torch.Tensor] = []
        self.sequences: List[torch.Tensor] = []

        self.total_distance_km = torch.zeros(
            batch_size, dtype=torch.float32, device=device
        )
        self.total_time_hours = torch.zeros(
            batch_size, dtype=torch.float32, device=device
        )


class VRPTWEnvironment:
    """Batched VRPTW env used identically for training and CG pricing."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        demands: torch.Tensor,
        coords: torch.Tensor,
        tw_start: torch.Tensor,
        tw_end: torch.Tensor,
        service_times: torch.Tensor,
        travel_time_matrix: torch.Tensor,
        distance_matrix_km: torch.Tensor,
        vehicle_capacity: float,
        depot_tw_end: float,
        site_mask: torch.Tensor,
        forbidden_arcs: Optional[torch.Tensor] = None,
        enforced_arcs: Optional[torch.Tensor] = None,
    ) -> VRPTWState:
        """Initialise a fresh state for one episode.

        All tensor arguments have shape (batch, …) where batch may equal
        the POMO expansion (original_batch * num_pomo_starts).

        Args
        ----
        demands            (B, N)
        coords             (B, N, 2)
        tw_start           (B, N)       hours
        tw_end             (B, N)       hours
        service_times      (B, N)       hours
        travel_time_matrix (B, N, N)    hours
        distance_matrix_km (B, N, N)    km
        vehicle_capacity   float
        depot_tw_end       float        hours
        site_mask          (B, N)       True = accessible
        forbidden_arcs     (B, N, N) bool  or None
        enforced_arcs      (B, N, N) bool  or None
        """
        self.demands = demands
        self.coords = coords
        self.tw_start = tw_start
        self.tw_end = tw_end
        self.service_times = service_times
        self.travel_time_matrix = travel_time_matrix
        self.distance_matrix_km = distance_matrix_km
        self.vehicle_capacity = vehicle_capacity
        self.depot_tw_end = depot_tw_end
        self.site_mask = site_mask
        self.forbidden_arcs = forbidden_arcs
        self.enforced_arcs = enforced_arcs

        batch_size = demands.shape[0]
        num_nodes = demands.shape[1]

        state = VRPTWState(batch_size, num_nodes, self.device)
        state.remaining_capacity.fill_(vehicle_capacity)
        state.current_time = tw_start[:, 0].clone()

        # Mark depot as "visited" so it is excluded from the standard
        # visited-node mask.  We control depot accessibility separately.
        state.visited[:, 0] = True

        return state

    # ------------------------------------------------------------------
    # Masking  (enforces ALL constraints simultaneously)
    # ------------------------------------------------------------------

    def get_action_mask(self, state: VRPTWState) -> torch.Tensor:
        """Return (B, N) bool mask.  True ⇒ node is *infeasible*.

        The mask enforces:
            1. Already-visited nodes
            2. Capacity
            3. Time-window (arrival ≤ tw_end)
            4. Return-to-depot feasibility after visit
            5. Site dependency
            6. Forbidden arcs  (B&B)
            7. Enforced arcs   (B&B)
            8. Depot availability
        """
        B = state.batch_size
        N = state.num_nodes
        batch_idx = torch.arange(B, device=self.device)

        mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)

        # 1 — already visited
        mask = mask | state.visited

        # 2 — capacity
        mask = mask | (self.demands > state.remaining_capacity.unsqueeze(1))

        # 3 — time-window reachability
        travel_from_cur = self.travel_time_matrix[batch_idx, state.current_node]
        arrival = state.current_time.unsqueeze(1) + travel_from_cur
        mask = mask | (arrival > self.tw_end)

        # 4 — must still be able to return to depot after serving
        effective_arrival = torch.max(arrival, self.tw_start)
        depart_after_service = effective_arrival + self.service_times
        travel_to_depot = self.travel_time_matrix[:, :, 0]
        depot_return_time = depart_after_service + travel_to_depot
        cannot_return = depot_return_time > self.depot_tw_end
        # apply only to customers, not depot itself
        mask[:, 1:] = mask[:, 1:] | cannot_return[:, 1:]

        # 5 — site dependency
        mask = mask | (~self.site_mask)

        # 6 — forbidden arcs
        if self.forbidden_arcs is not None:
            forbidden = self.forbidden_arcs[batch_idx, state.current_node]
            mask = mask | forbidden

        # 7 — enforced arcs
        if self.enforced_arcs is not None:
            enforced = self.enforced_arcs[batch_idx, state.current_node]
            has_enforced = enforced.any(dim=1)                      # (B,)
            # Where an enforcement exists, mask everything that is NOT
            # the enforced target (depot stays accessible as fallback).
            non_target = ~enforced
            non_target[:, 0] = False          # don't force-mask depot
            mask = torch.where(
                has_enforced.unsqueeze(1), mask | non_target, mask
            )

        # 8 — depot handling
        #   Before the first customer is visited ⇒ depot masked (no empty routes).
        #   After  at least one customer visited  ⇒ depot available.
        mask[:, 0] = ~state.route_started

        # 9 — safety: depot-escape-hatch
        # If a vehicle is trapped by B&B constraints, time windows, or site dependency
        # (all customers are masked), we MUST guarantee returning to the depot is unmasked
        # as a fail-safe, rather than forcing an illegal action.
        all_customers_masked = mask[:, 1:].all(dim=1)
        mask[all_customers_masked, 0] = False

        # 10 — finished routes: keep depot open as a harmless no-op
        mask[state.done] = True
        mask[state.done, 0] = False

        return mask

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: VRPTWState, action: torch.Tensor,
             log_prob: torch.Tensor) -> VRPTWState:
        """Execute actions and advance the state in-place.

        Args
        ----
        action   (B,) long   — selected node index
        log_prob (B,) float  — log-probability of the selected action
        """
        B = state.batch_size
        batch_idx = torch.arange(B, device=self.device)
        was_done = state.done.clone()

        # Travel metrics
        tt = self.travel_time_matrix[batch_idx, state.current_node, action]
        dd = self.distance_matrix_km[batch_idx, state.current_node, action]

        arrival = state.current_time + tt
        start_service = torch.max(arrival, self.tw_start[batch_idx, action])
        end_service = start_service + self.service_times[batch_idx, action]

        # Only update active (not-yet-done) instances
        active = ~was_done

        state.current_time = torch.where(active, end_service, state.current_time)
        state.remaining_capacity = torch.where(
            active,
            state.remaining_capacity - self.demands[batch_idx, action],
            state.remaining_capacity,
        )
        state.visited[batch_idx, action] = state.visited[batch_idx, action] | active
        state.current_node = torch.where(active, action, state.current_node)
        state.total_distance_km = torch.where(
            active, state.total_distance_km + dd, state.total_distance_km
        )
        state.total_time_hours = torch.where(
            active,
            state.current_time - self.tw_start[:, 0],
            state.total_time_hours,
        )

        is_customer = action > 0
        first_visit = is_customer & (~state.route_started) & active
        state.first_customer = torch.where(
            first_visit, action, state.first_customer
        )
        state.route_started = state.route_started | (is_customer & active)

        is_depot = (action == 0)
        # If we return to depot (or start at depot due to all customers being masked), we are done.
        state.done = state.done | (is_depot & active)

        state.log_probs.append(log_prob)
        state.sequences.append(action)

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_done(self, state: VRPTWState) -> bool:
        return state.done.all().item()
