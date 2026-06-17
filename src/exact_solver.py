"""Exact MIP solver for SD-VRPTW using OR-Tools SCIP backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from ortools.linear_solver import pywraplp

from src.data_loader import ProblemData


@dataclass
class ExactRoute:
    route_id: int
    vehicle_type: str
    customer_indices: List[int]
    total_cost: float
    total_distance_km: float
    total_time_hours: float


@dataclass
class ExactSolveResult:
    status: str
    objective_value: float
    best_bound: float
    mip_gap_percent: float
    solve_time_seconds: float
    selected_vehicle_count: int
    routes: List[ExactRoute]


def _is_arc_time_feasible(
    i: int,
    j: int,
    travel_time: np.ndarray,
    node_tw_start: np.ndarray,
    node_tw_end: np.ndarray,
    node_service: np.ndarray,
    depot_tw_start: float,
    depot_tw_end: float,
) -> bool:
    if i == j:
        return False
    if i == 0 and j == 0:
        return False

    earliest_departure = depot_tw_start if i == 0 else node_tw_start[i] + node_service[i]
    earliest_arrival = earliest_departure + travel_time[i, j]

    if j == 0:
        return earliest_arrival <= depot_tw_end + 1e-8

    if earliest_arrival > node_tw_end[j] + 1e-8:
        return False

    earliest_finish_at_j = max(earliest_arrival, node_tw_start[j]) + node_service[j]
    earliest_return_to_depot = earliest_finish_at_j + travel_time[j, 0]
    return earliest_return_to_depot <= depot_tw_end + 1e-8


def _extract_route_from_vehicle(
    vehicle_index: int,
    x: Dict[Tuple[int, int, int], pywraplp.Variable],
    outgoing_nodes: List[List[int]],
) -> List[int]:
    route_nodes: List[int] = []
    current_node = 0
    visited_nodes = set()
    max_steps = max(2, len(outgoing_nodes) + 2)

    for _ in range(max_steps):
        candidates = outgoing_nodes[current_node]
        next_node: Optional[int] = None
        best_value = -1.0
        for candidate_node in candidates:
            variable = x.get((vehicle_index, current_node, candidate_node))
            if variable is None:
                continue
            value = variable.solution_value()
            if value > best_value:
                best_value = value
                next_node = candidate_node

        if next_node is None or best_value < 0.5:
            break
        if next_node == 0:
            break
        if next_node in visited_nodes:
            break

        route_nodes.append(next_node)
        visited_nodes.add(next_node)
        current_node = next_node

    return route_nodes


def solve_exact_sdvrptw(
    problem: ProblemData,
    time_limit_seconds: int = 120,
    assume_unlimited_vehicles: bool = True,
    max_vehicles_per_type: Optional[int] = None,
    solver_name: str = "SCIP",
    verbose: bool = False,
) -> ExactSolveResult:
    """Solve heterogeneous SD-VRPTW exactly with MIP."""
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Could not create {solver_name} solver.")
    solver.SetTimeLimit(int(time_limit_seconds * 1000))
    if verbose:
        solver.EnableOutput()

    num_customers = problem.num_customers
    num_nodes = num_customers + 1
    depot = 0

    node_demand = np.zeros(num_nodes, dtype=np.float64)
    node_demand[1:] = problem.demands
    node_tw_start = np.zeros(num_nodes, dtype=np.float64)
    node_tw_end = np.full(num_nodes, problem.depot_tw_end, dtype=np.float64)
    node_tw_start[1:] = problem.tw_start
    node_tw_end[1:] = problem.tw_end
    node_service = np.zeros(num_nodes, dtype=np.float64)
    node_service[1:] = problem.service_times
    distance_km = problem.distance_matrix_meters / 1000.0

    vehicle_types = list(problem.vehicle_types)
    vehicle_type_index = {vehicle_type: idx for idx, vehicle_type in enumerate(vehicle_types)}

    physical_vehicles: List[Tuple[str, int]] = []
    for vehicle_type in vehicle_types:
        if assume_unlimited_vehicles:
            count = num_customers
        else:
            count = int(problem.vehicle_count.get(vehicle_type, 999999))
        if max_vehicles_per_type is not None:
            count = min(count, int(max_vehicles_per_type))
        count = min(count, num_customers)
        for local_id in range(count):
            physical_vehicles.append((vehicle_type, local_id))

    num_vehicles = len(physical_vehicles)
    if num_vehicles == 0:
        return ExactSolveResult(
            status="NO_VEHICLES",
            objective_value=float("inf"),
            best_bound=float("inf"),
            mip_gap_percent=float("inf"),
            solve_time_seconds=0.0,
            selected_vehicle_count=0,
            routes=[],
        )

    vehicle_capacity = []
    vehicle_fixed_cost = []
    vehicle_cost_per_km = []
    vehicle_cost_per_hour = []
    vehicle_travel_time = []
    for vehicle_type, _ in physical_vehicles:
        vehicle_capacity.append(problem.vehicle_capacity[vehicle_type])
        vehicle_fixed_cost.append(problem.vehicle_fixed_cost[vehicle_type])
        vehicle_cost_per_km.append(problem.vehicle_cost_per_km[vehicle_type])
        vehicle_cost_per_hour.append(problem.vehicle_cost_per_hour[vehicle_type])
        vehicle_travel_time.append(problem.travel_time_matrices[vehicle_type])

    maximum_travel_time = max(float(np.max(matrix)) for matrix in vehicle_travel_time)
    maximum_service_time = float(np.max(node_service)) if num_customers > 0 else 0.0
    big_m = problem.depot_tw_end + maximum_travel_time + maximum_service_time + 1.0

    x: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
    outgoing_nodes: List[List[List[int]]] = []
    incoming_nodes: List[List[List[int]]] = []
    feasible_arcs_per_vehicle: List[List[Tuple[int, int]]] = []

    for vehicle in range(num_vehicles):
        vehicle_type, _ = physical_vehicles[vehicle]
        vehicle_type_id = vehicle_type_index[vehicle_type]
        travel_time = vehicle_travel_time[vehicle]

        vehicle_outgoing = [[] for _ in range(num_nodes)]
        vehicle_incoming = [[] for _ in range(num_nodes)]
        vehicle_arcs: List[Tuple[int, int]] = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if i > 0 and not problem.site_dependency[i - 1, vehicle_type_id]:
                    continue
                if j > 0 and not problem.site_dependency[j - 1, vehicle_type_id]:
                    continue
                if not _is_arc_time_feasible(
                    i=i,
                    j=j,
                    travel_time=travel_time,
                    node_tw_start=node_tw_start,
                    node_tw_end=node_tw_end,
                    node_service=node_service,
                    depot_tw_start=problem.depot_tw_start,
                    depot_tw_end=problem.depot_tw_end,
                ):
                    continue

                x[(vehicle, i, j)] = solver.BoolVar(f"x_{vehicle}_{i}_{j}")
                vehicle_outgoing[i].append(j)
                vehicle_incoming[j].append(i)
                vehicle_arcs.append((i, j))

        outgoing_nodes.append(vehicle_outgoing)
        incoming_nodes.append(vehicle_incoming)
        feasible_arcs_per_vehicle.append(vehicle_arcs)

    use_vehicle = [solver.BoolVar(f"use_vehicle_{vehicle}") for vehicle in range(num_vehicles)]
    visit = {}
    for vehicle in range(num_vehicles):
        for i in range(1, num_nodes):
            visit[(vehicle, i)] = solver.BoolVar(f"visit_{vehicle}_{i}")

    service_start = {}
    for vehicle in range(num_vehicles):
        service_start[(vehicle, depot)] = solver.NumVar(
            problem.depot_tw_start, problem.depot_tw_start, f"t_{vehicle}_0"
        )
        for i in range(1, num_nodes):
            service_start[(vehicle, i)] = solver.NumVar(
                0.0, problem.depot_tw_end, f"t_{vehicle}_{i}"
            )

    return_time = [
        solver.NumVar(0.0, problem.depot_tw_end, f"return_{vehicle}")
        for vehicle in range(num_vehicles)
    ]

    for i in range(1, num_nodes):
        solver.Add(
            sum(visit[(vehicle, i)] for vehicle in range(num_vehicles)) == 1
        )

    for vehicle in range(num_vehicles):
        for i in range(1, num_nodes):
            incoming = sum(x[(vehicle, j, i)] for j in incoming_nodes[vehicle][i])
            outgoing = sum(x[(vehicle, i, j)] for j in outgoing_nodes[vehicle][i])
            solver.Add(incoming == visit[(vehicle, i)])
            solver.Add(outgoing == visit[(vehicle, i)])

        solver.Add(
            sum(x[(vehicle, depot, j)] for j in outgoing_nodes[vehicle][depot]) == use_vehicle[vehicle]
        )
        solver.Add(
            sum(x[(vehicle, i, depot)] for i in incoming_nodes[vehicle][depot]) == use_vehicle[vehicle]
        )

        solver.Add(
            sum(node_demand[i] * visit[(vehicle, i)] for i in range(1, num_nodes))
            <= vehicle_capacity[vehicle]
        )

        vehicle_type, _ = physical_vehicles[vehicle]
        type_index = vehicle_type_index[vehicle_type]
        for i in range(1, num_nodes):
            if not problem.site_dependency[i - 1, type_index]:
                solver.Add(visit[(vehicle, i)] == 0)

        for i in range(1, num_nodes):
            solver.Add(
                service_start[(vehicle, i)]
                >= node_tw_start[i] - big_m * (1 - visit[(vehicle, i)])
            )
            solver.Add(
                service_start[(vehicle, i)]
                <= node_tw_end[i] + big_m * (1 - visit[(vehicle, i)])
            )

        travel_time = vehicle_travel_time[vehicle]
        for j in range(1, num_nodes):
            if (vehicle, depot, j) not in x:
                continue
            solver.Add(
                service_start[(vehicle, j)]
                >= problem.depot_tw_start + travel_time[depot, j]
                - big_m * (1 - x[(vehicle, depot, j)])
            )

        for i in range(1, num_nodes):
            for j in range(1, num_nodes):
                if i == j:
                    continue
                if (vehicle, i, j) not in x:
                    continue
                solver.Add(
                    service_start[(vehicle, j)]
                    >= service_start[(vehicle, i)]
                    + node_service[i]
                    + travel_time[i, j]
                    - big_m * (1 - x[(vehicle, i, j)])
                )

        for i in range(1, num_nodes):
            if (vehicle, i, depot) not in x:
                continue
            solver.Add(
                return_time[vehicle]
                >= service_start[(vehicle, i)]
                + node_service[i]
                + travel_time[i, depot]
                - big_m * (1 - x[(vehicle, i, depot)])
            )
        solver.Add(return_time[vehicle] <= problem.depot_tw_end * use_vehicle[vehicle])

    objective = solver.Objective()
    for vehicle in range(num_vehicles):
        objective.SetCoefficient(use_vehicle[vehicle], vehicle_fixed_cost[vehicle])
        objective.SetCoefficient(return_time[vehicle], vehicle_cost_per_hour[vehicle])
        for i, j in feasible_arcs_per_vehicle[vehicle]:
            arc_distance = distance_km[i, j]
            objective.SetCoefficient(
                x[(vehicle, i, j)],
                vehicle_cost_per_km[vehicle] * arc_distance,
            )
    objective.SetMinimization()

    status_code = solver.Solve()
    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status = status_map.get(status_code, f"STATUS_{status_code}")

    objective_value = float("inf")
    best_bound = float("inf")
    gap_percent = float("inf")
    selected_vehicle_count = 0
    routes: List[ExactRoute] = []
    if status in {"OPTIMAL", "FEASIBLE"}:
        objective_value = objective.Value()
        if hasattr(objective, "BestBound"):
            best_bound = objective.BestBound()
            if abs(objective_value) > 1e-8:
                gap_percent = max(
                    0.0, 100.0 * (objective_value - best_bound) / abs(objective_value)
                )
            else:
                gap_percent = 0.0
        for vehicle in range(num_vehicles):
            if use_vehicle[vehicle].solution_value() <= 0.5:
                continue
            customer_nodes = _extract_route_from_vehicle(
                vehicle_index=vehicle,
                x=x,
                outgoing_nodes=outgoing_nodes[vehicle],
            )
            customer_indices = [node_id - 1 for node_id in customer_nodes if node_id > 0]
            if not customer_indices:
                continue

            total_distance_km = 0.0
            previous_node = 0
            for current_customer in customer_nodes:
                total_distance_km += float(distance_km[previous_node, current_customer])
                previous_node = current_customer
            total_distance_km += float(distance_km[previous_node, 0])

            total_time_hours = float(return_time[vehicle].solution_value())
            vehicle_type, _ = physical_vehicles[vehicle]
            route_cost = (
                vehicle_fixed_cost[vehicle]
                + vehicle_cost_per_km[vehicle] * total_distance_km
                + vehicle_cost_per_hour[vehicle] * total_time_hours
            )
            routes.append(
                ExactRoute(
                    route_id=len(routes),
                    vehicle_type=vehicle_type,
                    customer_indices=customer_indices,
                    total_cost=route_cost,
                    total_distance_km=total_distance_km,
                    total_time_hours=total_time_hours,
                )
            )

        selected_vehicle_count = len(routes)

    return ExactSolveResult(
        status=status,
        objective_value=objective_value,
        best_bound=best_bound,
        mip_gap_percent=gap_percent,
        solve_time_seconds=solver.WallTime() / 1000.0,
        selected_vehicle_count=selected_vehicle_count,
        routes=routes,
    )
