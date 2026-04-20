"""Label Setting Algorithm (ESPPRC) for SD-VRPTW pricing subproblem.

Memory-safe implementation with:
    - Aggressive domination rules using bitmask visited sets
    - Beam width cap per node to prevent combinatorial explosion
    - Hard cap on total labels created
    - Explicit gc.collect() after solving
"""

import gc
import heapq
import logging
from typing import Dict, List, Set, Tuple

import numpy as np

from src.data_loader import ProblemData

logger = logging.getLogger(__name__)


def solve_espprc(
    problem: ProblemData,
    dual_values: np.ndarray,
    vehicle_type: str,
    vehicle_idx: int,
    forbidden_arcs: Set[Tuple[int, int]],
    enforced_arcs: Set[Tuple[int, int]],
    vehicle_dual: float = 0.0,
    beam_width: int = 50,
    max_total_labels: int = 10000,
) -> List[Tuple[List[int], float, float, float, float]]:
    """Solve ESPPRC via label setting for a single vehicle type.

    Uses bitmask representation for visited sets (supports up to 63 customers).
    Labels are processed chronologically via a min-heap on time.

    Returns
    -------
    List of (customer_indices, total_cost, dist_km, time_h, reduced_cost)
    for routes with negative reduced cost, sorted by reduced cost ascending.
    """
    num_customers = problem.num_customers
    num_nodes = num_customers + 1

    if num_customers > 63:
        logger.error("DP solver bitmask supports at most 63 customers (got %d).", num_customers)
        return []

    capacity = problem.vehicle_capacity[vehicle_type]
    fixed_cost = problem.vehicle_fixed_cost[vehicle_type]
    cost_per_km = problem.vehicle_cost_per_km[vehicle_type]
    cost_per_hour = problem.vehicle_cost_per_hour[vehicle_type]
    depot_tw_start = problem.depot_tw_start
    depot_tw_end = problem.depot_tw_end

    tt_mat = problem.travel_time_matrices[vehicle_type]
    dist_km_mat = problem.distance_matrix_meters / 1000.0

    accessible = [
        c for c in range(num_customers)
        if problem.site_dependency[c, vehicle_idx]
    ]
    if not accessible:
        return []

    enforced_map: Dict[int, int] = {}
    for u, v in (enforced_arcs or set()):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            enforced_map[u] = v

    forbidden_set = set(forbidden_arcs) if forbidden_arcs else set()

    # --- Label representation ---
    # Heap entry: (time, label_id, node, dist_km, load, dual_sum, visited_bitmask, path_tuple)
    label_counter = 0
    initial = (depot_tw_start, label_counter, 0, 0.0, 0.0, 0.0, 0, ())
    label_counter += 1

    queue: list = [initial]

    # Stored labels per node for domination: list of (time, dist, load, dual_sum, visited_bitmask)
    stored_labels: Dict[int, list] = {n: [] for n in range(num_nodes)}
    stored_labels[0].append((depot_tw_start, 0.0, 0.0, 0.0, 0))

    results: List[Tuple[List[int], float, float, float, float]] = []
    total_created = 0

    while queue and total_created < max_total_labels:
        cur_time, _, cur_node, cur_dist, cur_load, cur_duals, cur_vis, cur_path = heapq.heappop(queue)

        enforced_next = enforced_map.get(cur_node)

        for c in accessible:
            if cur_vis & (1 << c):
                continue

            next_node = c + 1

            if (cur_node, next_node) in forbidden_set:
                continue

            if enforced_next is not None and next_node != enforced_next:
                continue

            new_load = cur_load + problem.demands[c]
            if new_load > capacity:
                continue

            travel_time = tt_mat[cur_node, next_node]
            arrival = cur_time + travel_time
            if arrival > problem.tw_end[c]:
                continue

            start_svc = max(arrival, problem.tw_start[c])
            end_svc = start_svc + problem.service_times[c]

            if end_svc + tt_mat[next_node, 0] > depot_tw_end:
                continue

            new_dist = cur_dist + dist_km_mat[cur_node, next_node]
            new_time = end_svc
            new_duals = cur_duals + dual_values[c]
            new_vis = cur_vis | (1 << c)
            new_path = cur_path + (c,)

            is_dominated = _check_dominated_and_prune(
                stored_labels[next_node], new_time, new_dist,
                new_load, new_duals, new_vis,
                cost_per_km, cost_per_hour, depot_tw_start, beam_width,
            )
            if is_dominated:
                continue

            stored_labels[next_node].append(
                (new_time, new_dist, new_load, new_duals, new_vis)
            )

            heap_entry = (
                new_time, label_counter, next_node,
                new_dist, new_load, new_duals, new_vis, new_path,
            )
            label_counter += 1
            heapq.heappush(queue, heap_entry)
            total_created += 1

        # Try completing route: return to depot
        if cur_node != 0 and cur_path:
            if (cur_node, 0) not in forbidden_set:
                return_tt = tt_mat[cur_node, 0]
                return_dist = dist_km_mat[cur_node, 0]
                final_time = cur_time + return_tt

                if final_time <= depot_tw_end:
                    total_dist = cur_dist + return_dist
                    total_time_h = final_time - depot_tw_start
                    total_cost = (
                        fixed_cost
                        + cost_per_km * total_dist
                        + cost_per_hour * total_time_h
                    )
                    rc = total_cost - cur_duals - vehicle_dual

                    if rc < -1e-6:
                        results.append((
                            list(cur_path), total_cost,
                            total_dist, total_time_h, rc,
                        ))

    if total_created >= max_total_labels:
        logger.warning(
            "DP solver hit label limit (%d) for %s.", max_total_labels, vehicle_type,
        )

    gc.collect()
    results.sort(key=lambda x: x[4])
    return results[:beam_width]


def _check_dominated_and_prune(
    stored: list,
    new_time: float,
    new_dist: float,
    new_load: float,
    new_duals: float,
    new_vis: int,
    cost_per_km: float,
    cost_per_hour: float,
    depot_tw_start: float,
    beam_width: int,
) -> bool:
    """Check if the new label is dominated. Also prune labels dominated by it.

    Returns True if the new label is dominated (should be discarded).
    """
    surviving = []
    is_dominated = False

    for stored_label in stored:
        s_time, s_dist, s_load, s_duals, s_vis = stored_label

        # stored dominates new?  (bitmask subset: s_vis ⊆ new_vis iff s_vis & ~new_vis == 0)
        if (s_time <= new_time + 1e-10
                and s_dist <= new_dist + 1e-10
                and s_load <= new_load + 1e-10
                and s_duals >= new_duals - 1e-10
                and (s_vis & ~new_vis) == 0):
            is_dominated = True
            surviving.append(stored_label)
            continue

        # new dominates stored?
        if (new_time <= s_time + 1e-10
                and new_dist <= s_dist + 1e-10
                and new_load <= s_load + 1e-10
                and new_duals >= s_duals - 1e-10
                and (new_vis & ~s_vis) == 0):
            continue  # drop the stored label

        surviving.append(stored_label)

    stored.clear()
    stored.extend(surviving)

    if is_dominated:
        return True

    # Enforce beam width
    if len(stored) >= beam_width:
        def _rc_estimate(label):
            t, d, _, ds, _ = label
            return cost_per_km * d + cost_per_hour * (t - depot_tw_start) - ds

        stored.sort(key=_rc_estimate)
        del stored[beam_width - 1:]

    return False
