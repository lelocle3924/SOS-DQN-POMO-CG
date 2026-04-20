"""Restricted Master Problem (Set Covering) solved with OR-Tools GLOP / SCIP."""

import gc
import logging
from typing import Optional, List, Dict

import numpy as np
from ortools.linear_solver import pywraplp

from src.column_pool import ColumnPool
from src.data_loader import ProblemData

logger = logging.getLogger(__name__)


class MasterProblemResult:
    """Container for the RMP LP or IP solution."""

    def __init__(self) -> None:
        self.objective_value: float = float("inf")
        self.route_weights: np.ndarray = np.array([])
        self.dual_values: np.ndarray = np.array([])
        self.vehicle_dual_values: Dict[str, float] = {}
        self.status: str = "NOT_SOLVED"


# ---------------------------------------------------------------------------
# LP Relaxation  (called at every CG iteration)
# ---------------------------------------------------------------------------

def solve_master_problem(
    column_pool: ColumnPool,
    problem: ProblemData,
    valid_column_indices: Optional[List[int]] = None,
) -> MasterProblemResult:
    """
    Solve the LP relaxation of the Set Covering RMP:

        min  sum_k  c_k * theta_k
        s.t. sum_k  a_{ik} * theta_k  >= 1   for every customer i
             sum_{k in V_v} theta_k <= F_v   for every vehicle type v
             theta_k >= 0
    """
    result = MasterProblemResult()
    num_customers = problem.num_customers
    coverage = column_pool.get_coverage_matrix(num_customers)
    costs = column_pool.get_cost_vector()
    num_routes = column_pool.num_routes

    if num_routes == 0:
        logger.warning("Column pool is empty — cannot solve RMP.")
        result.status = "NO_COLUMNS"
        return result

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("Could not instantiate the GLOP solver.")

    # Decision variables
    theta = []
    for k in range(num_routes):
        # If valid_column_indices is provided, force invalid columns to 0
        ub = solver.infinity()
        if valid_column_indices is not None and k not in valid_column_indices:
            ub = 0.0
        theta.append(solver.NumVar(0.0, ub, f"theta_{k}"))

    # Artificial variables to guarantee feasibility
    artificial_vars = []
    for i in range(num_customers):
        artificial_vars.append(solver.NumVar(0.0, solver.infinity(), f"artificial_{i}"))

    # Set-covering constraints  (one per customer)
    customer_constraints = []
    for i in range(num_customers):
        ct = solver.Constraint(1.0, solver.infinity(), f"cover_{i}")
        for k in range(num_routes):
            if coverage[i, k] > 0.5:
                ct.SetCoefficient(theta[k], coverage[i, k])
        ct.SetCoefficient(artificial_vars[i], 1.0)
        customer_constraints.append(ct)

    # Objective
    objective = solver.Objective()
    for k in range(num_routes):
        objective.SetCoefficient(theta[k], costs[k])
    for i in range(num_customers):
        objective.SetCoefficient(artificial_vars[i], 100000.0)
    objective.SetMinimization()

    # Vehicle count constraints
    vehicle_constraints = {}
    for vtype in problem.vehicle_types:
        max_count = problem.vehicle_count.get(vtype, 999999)
        if max_count < 999999:
            ct = solver.Constraint(-solver.infinity(), max_count, f"fleet_{vtype}")
            for k, route in enumerate(column_pool.routes):
                if route.vehicle_type == vtype:
                    ct.SetCoefficient(theta[k], 1.0)
            vehicle_constraints[vtype] = ct

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        result.status = "OPTIMAL"
        result.objective_value = objective.Value()
        result.route_weights = np.array([v.solution_value() for v in theta])
        result.dual_values = np.array(
            [c.dual_value() for c in customer_constraints]
        )
        for vtype, ct in vehicle_constraints.items():
            result.vehicle_dual_values[vtype] = ct.dual_value()
        logger.info(
            "RMP LP solved — obj=%.4f, active_cols=%d",
            result.objective_value,
            int((result.route_weights > 1e-8).sum()),
        )
    elif status == pywraplp.Solver.INFEASIBLE:
        result.status = "INFEASIBLE"
        logger.warning("RMP LP is infeasible.")
    else:
        result.status = f"SOLVER_STATUS_{status}"
        logger.warning("RMP LP returned unexpected status: %s", status)

    del solver
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Integer Master  (called once at the end for the final solution)
# ---------------------------------------------------------------------------

def solve_integer_master(
    column_pool: ColumnPool,
    problem: ProblemData,
    valid_column_indices: Optional[List[int]] = None,
) -> MasterProblemResult:
    """Solve the integer (binary) Set Covering master over all generated columns."""
    result = MasterProblemResult()
    num_customers = problem.num_customers
    coverage = column_pool.get_coverage_matrix(num_customers)
    costs = column_pool.get_cost_vector()
    num_routes = column_pool.num_routes

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("Could not instantiate the SCIP solver.")

    theta = []
    for k in range(num_routes):
        ub = 1
        if valid_column_indices is not None and k not in valid_column_indices:
            ub = 0
        theta.append(solver.IntVar(0, ub, f"theta_{k}"))

    artificial_vars = []
    for i in range(num_customers):
        artificial_vars.append(solver.IntVar(0, 1, f"artificial_{i}"))

    for i in range(num_customers):
        ct = solver.Constraint(1.0, solver.infinity(), f"cover_{i}")
        for k in range(num_routes):
            if coverage[i, k] > 0.5:
                ct.SetCoefficient(theta[k], coverage[i, k])
        ct.SetCoefficient(artificial_vars[i], 1.0)

    objective = solver.Objective()
    for k in range(num_routes):
        objective.SetCoefficient(theta[k], costs[k])
    for i in range(num_customers):
        objective.SetCoefficient(artificial_vars[i], 100000.0)
    objective.SetMinimization()

    # Vehicle count constraints
    for vtype in problem.vehicle_types:
        max_count = problem.vehicle_count.get(vtype, 999999)
        if max_count < 999999:
            ct = solver.Constraint(-solver.infinity(), max_count, f"fleet_{vtype}")
            for k, route in enumerate(column_pool.routes):
                if route.vehicle_type == vtype:
                    ct.SetCoefficient(theta[k], 1.0)

    status = solver.Solve()

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        result.status = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
        result.objective_value = objective.Value()
        result.route_weights = np.array([v.solution_value() for v in theta])
        result.dual_values = np.array([])
        logger.info(
            "Integer master solved — obj=%.4f, routes_selected=%d",
            result.objective_value,
            int((result.route_weights > 0.5).sum()),
        )
    else:
        result.status = f"SOLVER_STATUS_{status}"
        logger.warning("Integer master returned status: %s", status)

    del solver
    gc.collect()
    return result
