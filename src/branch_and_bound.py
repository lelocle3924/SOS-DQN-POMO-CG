"""Branch-and-Price with Feillet's edge-branching rule.

Branching is on the *arc* variables x_{ij}, NOT on route variables theta_k.
Edge flow is computed as  f_{ij} = sum_k b_{ijk} * theta_k ,
where b_{ijk} = 1 iff route k traverses arc (i,j).
"""

import gc
import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.column_pool import ColumnPool, Route
from src.column_selection.base import AbstractColumnSelector, ColumnSelectionState
from src.data_loader import ProblemData
from src.master_problem import (
    MasterProblemResult,
    solve_integer_master,
    solve_master_problem,
)
from src.pricing_orchestrator import PricingOrchestrator

logger = logging.getLogger(__name__)


# ======================================================================
# B&B node
# ======================================================================

@dataclass(order=False)
class BranchNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    lower_bound: float
    forbidden_arcs: Set[Tuple[int, int]] = field(default_factory=set)
    enforced_arcs: Set[Tuple[int, int]] = field(default_factory=set)
    status: str = "open"

    def __lt__(self, other: "BranchNode") -> bool:
        """Best-first: smallest lower bound first."""
        return self.lower_bound < other.lower_bound


# ======================================================================
# Edge-flow helpers
# ======================================================================

def get_valid_columns(
    column_pool: ColumnPool,
    forbidden_arcs: Set[Tuple[int, int]],
    enforced_arcs: Set[Tuple[int, int]],
) -> List[int]:
    """Return indices of columns that respect the branching constraints."""
    valid_indices = []
    for k, route in enumerate(column_pool.routes):
        seq = [0] + [c + 1 for c in route.visit_sequence] + [0]
        arcs = set(zip(seq[:-1], seq[1:]))
        
        # Check forbidden
        if any(arc in forbidden_arcs for arc in arcs):
            continue
            
        # Check enforced
        is_valid = True
        for u, v in enforced_arcs:
            if u in seq[:-1]:
                idx = seq.index(u)
                if seq[idx+1] != v:
                    is_valid = False
                    break
            if v in seq[1:]:
                idx = seq.index(v)
                if seq[idx-1] != u:
                    is_valid = False
                    break
        
        if is_valid:
            valid_indices.append(k)
            
    return valid_indices


def compute_edge_flows(
    column_pool: ColumnPool,
    route_weights: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """f_{ij} = sum_k  b_{ijk} * theta_k    (num_nodes x num_nodes)."""
    flows = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    max_routes = min(len(column_pool.routes), len(route_weights))
    for k, route in enumerate(column_pool.routes[:max_routes]):
        w = route_weights[k]
        if w < 1e-10:
            continue
        # arc sequence: depot -> c1+1 -> c2+1 -> … -> depot
        seq = [0] + [c + 1 for c in route.visit_sequence] + [0]
        for step in range(len(seq) - 1):
            flows[seq[step], seq[step + 1]] += w

    return flows


def select_branching_edge(
    edge_flows: np.ndarray,
    tolerance: float = 0.01,
) -> Optional[Tuple[int, int]]:
    """Pick the arc whose flow is closest to 0.5 (most fractional)."""
    N = edge_flows.shape[0]
    best_edge: Optional[Tuple[int, int]] = None
    best_score = 0.0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            frac_part = edge_flows[i, j] - int(edge_flows[i, j])
            score = min(frac_part, 1.0 - frac_part)
            if score > tolerance and score > best_score:
                best_score = score
                best_edge = (i, j)

    return best_edge


# ======================================================================
# Column-generation loop (at a single B&B node)
# ======================================================================

def _run_column_generation(
    problem: ProblemData,
    orchestrator: PricingOrchestrator,
    column_pool: ColumnPool,
    config: Dict,
    forbidden_arcs: Set[Tuple[int, int]],
    enforced_arcs: Set[Tuple[int, int]],
    column_selector: Optional[AbstractColumnSelector] = None,
) -> MasterProblemResult:
    max_iter = config["column_generation"]["max_cg_iterations"]

    for it in range(max_iter):
        valid_indices = get_valid_columns(column_pool, forbidden_arcs, enforced_arcs)
        rmp = solve_master_problem(column_pool, problem, valid_indices)
        if rmp.status != "OPTIMAL":
            logger.warning("CG iter %d: RMP status = %s", it, rmp.status)
            return rmp

        logger.info(
            "CG iter %d: obj=%.4f  cols=%d",
            it, rmp.objective_value, column_pool.num_routes,
        )

        bb_constraints = {
            "forbidden_arcs": forbidden_arcs,
            "enforced_arcs": enforced_arcs
        }
        
        #>>> GENERATE COLUMNS
        candidate_routes = orchestrator.generate_columns(
            problem, rmp.dual_values, rmp.vehicle_dual_values, bb_constraints
        )

        #>>> SELECT COLUMNS
        if column_selector is not None:
            selection_state = ColumnSelectionState(
                problem=problem,
                column_pool=column_pool,
                rmp_result=rmp,
                iteration_index=it,
                forbidden_arcs=forbidden_arcs,
                enforced_arcs=enforced_arcs,
                config=config,
            )
            new_routes = column_selector.select_columns(selection_state, candidate_routes)
        else:
            new_routes = candidate_routes
        
        added_count = 0
        for route in new_routes:
            added = column_pool.add_route(
                vehicle_type=route.vehicle_type,
                customer_indices=route.customer_indices,
                total_cost=route.total_cost,
                total_distance_km=route.total_distance_km,
                total_time_hours=route.total_time_hours,
                reduced_cost=route.reduced_cost
            )
            if added is not None:
                added_count += 1

        if added_count == 0:
            logger.info("CG converged after %d iterations.", it + 1)
            break

        logger.info("  +%d new columns", added_count)

    return rmp


# ======================================================================
# Full Branch & Price
# ======================================================================

def branch_and_price(
    problem: ProblemData,
    orchestrator: PricingOrchestrator,
    column_pool: ColumnPool,
    config: Dict,
    column_selector: Optional[AbstractColumnSelector] = None,
) -> Dict:
    """Run the Branch-&-Price algorithm.

    Returns
    -------
    dict with keys: routes, total_cost, nodes_explored, total_columns
    """
    max_bb_nodes = config["branch_and_bound"]["max_nodes"]
    frac_tol = config["branch_and_bound"]["branching_fractional_tolerance"]
    num_nodes = problem.num_customers + 1

    root = BranchNode(
        node_id=0, parent_id=None, depth=0,
        lower_bound=-float("inf"),
    )
    open_list: List[BranchNode] = [root]
    best_ub = float("inf")
    best_solution: Optional[List[Route]] = None
    best_forbidden_arcs: Set[Tuple[int, int]] = set()
    best_enforced_arcs: Set[Tuple[int, int]] = set()
    next_id = 1
    explored = 0

    while open_list and explored < max_bb_nodes:
        node = heapq.heappop(open_list)

        if node.lower_bound >= best_ub - 1e-6:
            node.status = "pruned"
            continue

        explored += 1
        logger.info(
            "\n=== B&B node %d  depth=%d  LB=%.4f  UB=%.4f ===",
            node.node_id, node.depth, node.lower_bound, best_ub,
        )

        rmp = _run_column_generation(
            problem, orchestrator, column_pool, config,
            node.forbidden_arcs, node.enforced_arcs,
            column_selector=column_selector,
        )

        if rmp.status != "OPTIMAL":
            node.status = "infeasible"
            continue

        node.lower_bound = rmp.objective_value
        if node.lower_bound >= best_ub - 1e-6:
            node.status = "pruned"
            continue

        # Check integrality via edge flows
        flows = compute_edge_flows(column_pool, rmp.route_weights, num_nodes)
        branch_edge = select_branching_edge(flows, frac_tol)

        if branch_edge is None:
            # LP solution is integer
            if rmp.objective_value < best_ub:
                best_ub = rmp.objective_value
                best_solution = _extract_selected_routes(
                    column_pool, rmp.route_weights,
                )
                best_forbidden_arcs = set(node.forbidden_arcs)
                best_enforced_arcs = set(node.enforced_arcs)
                logger.info("  ★ new incumbent: cost=%.4f", best_ub)
            node.status = "optimal"
            continue

        i, j = branch_edge
        logger.info(
            "  branching on arc (%d,%d)  flow=%.4f", i, j, flows[i, j],
        )

        # Child 0: forbid arc (i, j)
        child0 = BranchNode(
            node_id=next_id, parent_id=node.node_id,
            depth=node.depth + 1, lower_bound=node.lower_bound,
            forbidden_arcs=node.forbidden_arcs | {(i, j)},
            enforced_arcs=set(node.enforced_arcs),
        )
        next_id += 1
        heapq.heappush(open_list, child0)

        # Child 1: enforce arc (i, j)
        child1 = BranchNode(
            node_id=next_id, parent_id=node.node_id,
            depth=node.depth + 1, lower_bound=node.lower_bound,
            forbidden_arcs=set(node.forbidden_arcs),
            enforced_arcs=node.enforced_arcs | {(i, j)},
        )
        next_id += 1
        heapq.heappush(open_list, child1)

        gc.collect()

    # Fallback: solve the integer master over the full column pool
    if best_solution is None:
        logger.info("No integer solution from B&B — solving integer master.")
        # For the final integer master, we should only use valid columns from the best node,
        # but since we didn't find any integer solution, we just solve it over all columns
        # (which corresponds to the root node constraints).
        int_res = solve_integer_master(column_pool, problem)
        if int_res.status in ("OPTIMAL", "FEASIBLE"):
            best_solution = _extract_selected_routes(
                column_pool, int_res.route_weights,
            )
            best_ub = int_res.objective_value

    return {
        "routes": best_solution or [],
        "total_cost": best_ub,
        "nodes_explored": explored,
        "total_columns": column_pool.num_routes,
        "forbidden_arcs": list(best_forbidden_arcs),
        "enforced_arcs": list(best_enforced_arcs),
    }


def _extract_selected_routes(
    pool: ColumnPool, weights: np.ndarray,
) -> List[Route]:
    # The LP/IP solver can return a weight vector whose length is smaller than
    # the current pool size when late-added columns are not part of that model.
    # Limit extraction to aligned indices to avoid out-of-bounds access.
    max_aligned_routes = min(len(pool.routes), len(weights))
    return [
        pool.routes[route_index]
        for route_index in range(max_aligned_routes)
        if weights[route_index] > 0.5
    ]
