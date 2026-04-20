"""Standalone solution validator.

Re-checks every constraint independently of the solver environment:
    1. Customer coverage  (every customer visited at least once)
    2. Vehicle capacity
    3. Time windows
    4. Depot return feasibility
    5. Site dependency

Usage:
    python validate_solution.py --solution results/<run>/raw_output.json
                                --config  configs/default_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_problem
from src.column_pool import ColumnPool
from src.master_problem import solve_master_problem

logger = logging.getLogger(__name__)


def validate_solution(solution_path: str, config_path: str) -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    with open(solution_path, encoding="utf-8") as fh:
        solution = json.load(fh)

    problem = load_problem(
        orders_path=os.path.join(PROJECT_ROOT, config["problem"]["orders_file"]),
        trucks_path=os.path.join(PROJECT_ROOT, config["problem"]["trucks_file"]),
        distance_matrix_path=os.path.join(PROJECT_ROOT, config["problem"]["distance_matrix_file"]),
        depot_id=config["problem"]["depot_id"],
    )

    all_valid = True
    covered: set = set()
    recomputed_cost = 0.0
    
    # For RMP reconstruction
    column_pool = ColumnPool()
    
    forbidden_arcs = set(tuple(a) for a in solution.get("forbidden_arcs", []))
    enforced_arcs = set(tuple(a) for a in solution.get("enforced_arcs", []))

    for rd in solution["routes"]:
        route_id = rd["route_id"]
        vtype = rd["vehicle_type"]
        cust_idx = rd["customer_indices"]

        logger.info("\n--- Route %d  [%s] ---", route_id, vtype)
        logger.info("  Customers: %s", cust_idx)

        if vtype not in problem.vehicle_types:
            logger.error("  FAIL: unknown vehicle type '%s'", vtype)
            all_valid = False
            continue

        v_idx = problem.vehicle_types.index(vtype)
        cap = problem.vehicle_capacity[vtype]
        fixed = problem.vehicle_fixed_cost[vtype]
        cpkm = problem.vehicle_cost_per_km[vtype]
        cphr = problem.vehicle_cost_per_hour[vtype]
        tt_mat = problem.travel_time_matrices[vtype]
        dk_mat = problem.distance_matrix_meters / 1000.0

        # --- capacity ---
        total_demand = sum(problem.demands[c] for c in cust_idx)
        if total_demand > cap + 1e-6:
            logger.error(
                "  FAIL capacity: %.1f > %.1f", total_demand, cap,
            )
            all_valid = False
        else:
            logger.info("  OK   capacity: %.1f / %.1f", total_demand, cap)

        # --- time windows & travel ---
        cur_node = 0
        cur_time = problem.depot_tw_start
        total_dist = 0.0

        for c in cust_idx:
            node = c + 1
            tt = tt_mat[cur_node, node]
            total_dist += dk_mat[cur_node, node]
            arrival = cur_time + tt

            if arrival > problem.tw_end[c] + 1e-6:
                logger.error(
                    "  FAIL TW cust %d (%s): arrival=%.2f > tw_end=%.2f",
                    c, problem.customer_ids[c], arrival, problem.tw_end[c],
                )
                all_valid = False
            else:
                logger.info(
                    "  OK   TW cust %d (%s): arrival=%.2f  window=[%.1f, %.1f]",
                    c, problem.customer_ids[c], arrival,
                    problem.tw_start[c], problem.tw_end[c],
                )

            start_svc = max(arrival, problem.tw_start[c])
            cur_time = start_svc + problem.service_times[c]
            cur_node = node

        # --- depot return ---
        total_dist += dk_mat[cur_node, 0]
        return_time = cur_time + tt_mat[cur_node, 0]
        if return_time > problem.depot_tw_end + 1e-6:
            logger.error(
                "  FAIL depot return: %.2f > %.1f",
                return_time, problem.depot_tw_end,
            )
            all_valid = False
        else:
            logger.info("  OK   depot return at %.2f", return_time)

        total_time = return_time - problem.depot_tw_start
        route_cost = fixed + cpkm * total_dist + cphr * total_time
        recomputed_cost += route_cost

        # --- site dependency ---
        for c in cust_idx:
            if not problem.site_dependency[c, v_idx]:
                logger.error(
                    "  FAIL site-dep: cust %d (%s) ✗ %s",
                    c, problem.customer_ids[c], vtype,
                )
                all_valid = False

        # --- branching integrity ---
        seq = [0] + [c + 1 for c in cust_idx] + [0]
        arcs = set(zip(seq[:-1], seq[1:]))
        
        for arc in arcs:
            if arc in forbidden_arcs:
                logger.error("  FAIL branching: route uses forbidden arc %s", arc)
                all_valid = False
                
        for u, v in enforced_arcs:
            if u in seq[:-1]:
                idx = seq.index(u)
                if seq[idx+1] != v:
                    logger.error("  FAIL branching: route visits %d but not followed by %d", u, v)
                    all_valid = False
            if v in seq[1:]:
                idx = seq.index(v)
                if seq[idx-1] != u:
                    logger.error("  FAIL branching: route visits %d but not preceded by %d", v, u)
                    all_valid = False

        covered.update(cust_idx)
        
        # Add to pool for RMP check
        column_pool.add_route(
            vehicle_type=vtype,
            customer_indices=cust_idx,
            total_cost=route_cost,
            total_distance_km=total_dist,
            total_time_hours=total_time,
        )

    # --- coverage ---
    all_custs = set(range(problem.num_customers))
    uncovered = all_custs - covered
    if uncovered:
        uncov_names = [problem.customer_ids[c] for c in sorted(uncovered)]
        logger.error("\nFAIL: %d uncovered customers: %s", len(uncov_names), uncov_names)
        all_valid = False
    else:
        logger.info("\nOK: all %d customers covered.", problem.num_customers)

    # --- fleet size ---
    vehicle_counts = {}
    for rd in solution["routes"]:
        vtype = rd["vehicle_type"]
        vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
        
    for vtype, count in vehicle_counts.items():
        max_count = problem.vehicle_count.get(vtype, 999999)
        if count > max_count:
            logger.error("\nFAIL: Fleet size exceeded for %s: %d > %d", vtype, count, max_count)
            all_valid = False
        else:
            if max_count < 999999:
                logger.info("OK: Fleet size for %s: %d / %d", vtype, count, max_count)

    # --- cost comparison ---
    reported = solution.get("total_cost", 0)
    logger.info("Reported cost  : %.4f", reported)
    logger.info("Recomputed cost: %.4f", recomputed_cost)
    if abs(reported - recomputed_cost) > 0.1:
        logger.warning(
            "Cost mismatch: |%.4f - %.4f| = %.4f",
            reported, recomputed_cost, abs(reported - recomputed_cost),
        )

    # --- RMP Lower Bound Verification ---
    logger.info("\n--- RMP Lower Bound Verification ---")
    rmp_res = solve_master_problem(column_pool, problem)
    if rmp_res.status != "OPTIMAL":
        logger.error("FAIL: Final RMP is not OPTIMAL (status: %s)", rmp_res.status)
        all_valid = False
    else:
        logger.info("RMP LP Objective : %.4f", rmp_res.objective_value)
        if abs(reported - rmp_res.objective_value) > 0.1:
            logger.warning(
                "RMP Objective mismatch with reported cost: |%.4f - %.4f| = %.4f",
                reported, rmp_res.objective_value, abs(reported - rmp_res.objective_value)
            )
        else:
            logger.info("OK: RMP Objective matches reported cost.")

    if all_valid:
        logger.info("\n=== VALIDATION PASSED ===")
    else:
        logger.error("\n=== VALIDATION FAILED ===")

    return all_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SD-VRPTW solution")
    parser.add_argument("--solution", required=True, help="raw_output.json path")
    parser.add_argument("--config", default="configs/default_config.yaml")
    args = parser.parse_args()
    ok = validate_solution(args.solution, args.config)
    sys.exit(0 if ok else 1)
