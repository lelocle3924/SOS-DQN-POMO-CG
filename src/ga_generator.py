"""Genetic Algorithm column generator for DQN training data.

Produces diverse feasible VRPTW columns (routes) by evolving populations
with Order Crossover (OX) and swap/relocate mutations while respecting
time-window, capacity, and site-dependency constraints.
"""

import gc
import random
from typing import List, Optional, Set, Tuple

import numpy as np

from src.data_loader import ProblemData


def evaluate_route(
    problem: ProblemData,
    customers: List[int],
    vehicle_type: str,
    vehicle_idx: int,
    dual_values: np.ndarray,
    vehicle_dual: float = 0.0,
) -> Tuple[bool, float, float, float, float]:
    """Evaluate a route for feasibility and exact cost.

    Returns (is_feasible, total_cost, dist_km, time_h, reduced_cost).
    """
    if not customers:
        return False, 0.0, 0.0, 0.0, 0.0

    capacity = problem.vehicle_capacity[vehicle_type]
    fixed_cost = problem.vehicle_fixed_cost[vehicle_type]
    cost_per_km = problem.vehicle_cost_per_km[vehicle_type]
    cost_per_hour = problem.vehicle_cost_per_hour[vehicle_type]
    tt_mat = problem.travel_time_matrices[vehicle_type]
    dist_km_mat = problem.distance_matrix_meters / 1000.0

    for c in customers:
        if not problem.site_dependency[c, vehicle_idx]:
            return False, 0.0, 0.0, 0.0, 0.0

    total_demand = sum(problem.demands[c] for c in customers)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0, 0.0

    cur_node = 0
    cur_time = problem.depot_tw_start
    total_dist = 0.0

    for c in customers:
        node = c + 1
        travel = tt_mat[cur_node, node]
        total_dist += dist_km_mat[cur_node, node]
        arrival = cur_time + travel

        if arrival > problem.tw_end[c]:
            return False, 0.0, 0.0, 0.0, 0.0

        start_svc = max(arrival, problem.tw_start[c])
        cur_time = start_svc + problem.service_times[c]
        cur_node = node

    total_dist += dist_km_mat[cur_node, 0]
    return_time = cur_time + tt_mat[cur_node, 0]

    if return_time > problem.depot_tw_end:
        return False, 0.0, 0.0, 0.0, 0.0

    total_time = return_time - problem.depot_tw_start
    total_cost = fixed_cost + cost_per_km * total_dist + cost_per_hour * total_time
    dual_sum = sum(dual_values[c] for c in customers)
    reduced_cost = total_cost - dual_sum - vehicle_dual

    return True, total_cost, total_dist, total_time, reduced_cost


def _order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Order Crossover (OX): preserves relative order from parent2 for genes
    not in the selected segment of parent1."""
    if len(parent1) <= 2:
        return list(parent1)

    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child: List[Optional[int]] = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    segment_set = set(child[start:end + 1])
    pos = (end + 1) % size
    for gene in parent2:
        if gene not in segment_set:
            child[pos] = gene
            pos = (pos + 1) % size

    return [g for g in child if g is not None]


def _mutate_swap(customers: List[int]) -> List[int]:
    if len(customers) < 2:
        return list(customers)
    result = list(customers)
    i, j = random.sample(range(len(result)), 2)
    result[i], result[j] = result[j], result[i]
    return result


def _mutate_relocate(customers: List[int]) -> List[int]:
    if len(customers) < 2:
        return list(customers)
    result = list(customers)
    idx = random.randrange(len(result))
    gene = result.pop(idx)
    insert_pos = random.randrange(len(result) + 1)
    result.insert(insert_pos, gene)
    return result


def _make_random_feasible_route(
    problem: ProblemData,
    accessible: List[int],
    vehicle_type: str,
    vehicle_idx: int,
    dual_values: np.ndarray,
    vehicle_dual: float,
    max_length: int,
    max_attempts: int = 10,
) -> Optional[Tuple[List[int], float]]:
    """Build a feasible route using greedy nearest-feasible construction."""
    tt_mat = problem.travel_time_matrices[vehicle_type]
    capacity = problem.vehicle_capacity[vehicle_type]

    for _ in range(max_attempts):
        target_len = random.randint(1, min(max_length, len(accessible)))
        route: List[int] = []
        remaining = set(accessible)
        cur_node = 0
        cur_time = problem.depot_tw_start
        cur_load = 0.0

        while len(route) < target_len and remaining:
            candidates = []
            for c in remaining:
                node = c + 1
                arr = cur_time + tt_mat[cur_node, node]
                if arr > problem.tw_end[c]:
                    continue
                if cur_load + problem.demands[c] > capacity:
                    continue
                start_svc = max(arr, problem.tw_start[c])
                depart = start_svc + problem.service_times[c]
                if depart + tt_mat[node, 0] > problem.depot_tw_end:
                    continue
                candidates.append((c, arr))

            if not candidates:
                break

            # Mix of greedy (nearest) and random to create diversity
            if random.random() < 0.5 and len(candidates) > 1:
                chosen_c, _ = random.choice(candidates)
            else:
                chosen_c, _ = min(candidates, key=lambda x: x[1])

            node = chosen_c + 1
            arrival = cur_time + tt_mat[cur_node, node]
            cur_time = max(arrival, problem.tw_start[chosen_c]) + problem.service_times[chosen_c]
            cur_load += problem.demands[chosen_c]
            cur_node = node
            route.append(chosen_c)
            remaining.discard(chosen_c)

        if route:
            ok, _, _, _, rc = evaluate_route(
                problem, route, vehicle_type, vehicle_idx, dual_values, vehicle_dual,
            )
            if ok:
                return route, rc

    return None


def generate_columns_ga(
    problem: ProblemData,
    dual_values: np.ndarray,
    vehicle_type: str,
    vehicle_idx: int,
    vehicle_dual: float = 0.0,
    population_size: int = 50,
    num_generations: int = 100,
    mutation_rate: float = 0.3,
    max_route_length: int = 20,
) -> List[Tuple[List[int], float, float, float, float]]:
    """Evolve a population of routes and return diverse feasible columns.

    Returns list of (customer_indices, cost, dist_km, time_h, reduced_cost).
    """
    accessible = [
        c for c in range(problem.num_customers)
        if problem.site_dependency[c, vehicle_idx]
    ]
    if not accessible:
        return []

    # --- Seed population ---
    population: List[Tuple[List[int], float]] = []
    for _ in range(population_size * 3):
        result = _make_random_feasible_route(
            problem, accessible, vehicle_type, vehicle_idx,
            dual_values, vehicle_dual, max_route_length,
        )
        if result is not None:
            population.append(result)
        if len(population) >= population_size:
            break

    if not population:
        return []

    # --- Evolutionary loop ---
    for _ in range(num_generations):
        offspring: List[Tuple[List[int], float]] = []

        # Tournament selection → crossover
        for _ in range(population_size // 2):
            candidates_a = random.sample(population, min(3, len(population)))
            candidates_b = random.sample(population, min(3, len(population)))
            parent_a = min(candidates_a, key=lambda x: x[1])
            parent_b = min(candidates_b, key=lambda x: x[1])

            child_custs = _order_crossover(parent_a[0], parent_b[0])
            child_custs = [c for c in child_custs if c in accessible]
            if child_custs:
                ok, cost, dist, time_h, rc = evaluate_route(
                    problem, child_custs, vehicle_type, vehicle_idx,
                    dual_values, vehicle_dual,
                )
                if ok:
                    offspring.append((child_custs, rc))

        # Mutation
        for custs, _ in list(offspring) + list(population):
            if random.random() < mutation_rate:
                mutated = _mutate_swap(custs) if random.random() < 0.5 else _mutate_relocate(custs)
                ok, cost, dist, time_h, rc = evaluate_route(
                    problem, mutated, vehicle_type, vehicle_idx,
                    dual_values, vehicle_dual,
                )
                if ok:
                    offspring.append((mutated, rc))

        # Elitism: merge old + new, keep best
        combined = population + offspring
        combined.sort(key=lambda x: x[1])
        population = combined[:population_size]

    # --- Deduplicate and evaluate final pool ---
    results: List[Tuple[List[int], float, float, float, float]] = []
    seen: Set[Tuple[int, ...]] = set()

    for custs, _ in population:
        key = tuple(sorted(custs))
        if key in seen:
            continue
        seen.add(key)

        ok, cost, dist, time_h, rc = evaluate_route(
            problem, custs, vehicle_type, vehicle_idx, dual_values, vehicle_dual,
        )
        if ok:
            results.append((custs, cost, dist, time_h, rc))

    gc.collect()
    return results
