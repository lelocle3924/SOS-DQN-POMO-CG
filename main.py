"""Main entry-point for the Branch-&-Price SD-VRPTW solver."""

import argparse
import logging
import os
import sys
import time
from typing import Tuple
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import fix_all_seeds, load_config
from src.data_loader import ProblemData, load_problem
from src.column_pool import ColumnPool
from src.branch_and_bound import branch_and_price
from src.pomo_model import POMOModel
from src.column_selection.factory import build_column_selector
from src.pricing_orchestrator import PricingOrchestrator
from src.run_manager import (
    append_to_master_csv,
    create_run_folder,
    save_raw_output,
    setup_logging,
    shadow_copy_config,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Greedy initial routes  (seeds the RMP)
# ======================================================================

def build_initial_routes(problem: ProblemData,
                         column_pool: ColumnPool) -> None:
    """Nearest-neighbour heuristic per vehicle type to warm-start the RMP."""
    for v_idx, vtype in enumerate(problem.vehicle_types):
        capacity = problem.vehicle_capacity[vtype]
        fixed_cost = problem.vehicle_fixed_cost[vtype]
        cost_per_km = problem.vehicle_cost_per_km[vtype]
        cost_per_hour = problem.vehicle_cost_per_hour[vtype]
        tt_mat = problem.travel_time_matrices[vtype]
        dist_km = problem.distance_matrix_meters / 1000.0

        accessible = [
            c for c in range(problem.num_customers)
            if problem.site_dependency[c, v_idx]
        ]
        unserved = set(accessible)

        while unserved:
            route_customers = []
            cur_node = 0
            rem_cap = capacity
            cur_time = problem.depot_tw_start
            total_dist = 0.0

            while True:
                best, best_arr = None, float("inf")
                for c in unserved:
                    node = c + 1
                    arr = cur_time + tt_mat[cur_node, node]
                    if problem.demands[c] > rem_cap:
                        continue
                    if arr > problem.tw_end[c]:
                        continue
                    svc_start = max(arr, problem.tw_start[c])
                    depart = svc_start + problem.service_times[c]
                    if depart + tt_mat[node, 0] > problem.depot_tw_end:
                        continue
                    if arr < best_arr:
                        best_arr = arr
                        best = c

                if best is None:
                    break

                node = best + 1
                total_dist += dist_km[cur_node, node]
                arr = cur_time + tt_mat[cur_node, node]
                cur_time = max(arr, problem.tw_start[best]) + problem.service_times[best]
                rem_cap -= problem.demands[best]
                cur_node = node
                route_customers.append(best)
                unserved.discard(best)

            if not route_customers:
                break

            total_dist += dist_km[cur_node, 0]
            total_time = cur_time - problem.depot_tw_start + tt_mat[cur_node, 0]
            total_cost = fixed_cost + cost_per_km * total_dist + cost_per_hour * total_time

            column_pool.add_route(
                vehicle_type=vtype,
                customer_indices=route_customers,
                total_cost=total_cost,
                total_distance_km=total_dist,
                total_time_hours=total_time,
            )

    logger.info("Greedy warm-start: %d initial routes.", column_pool.num_routes)


# ======================================================================
# POMO model loader
# ======================================================================

def load_pomo_model(config: dict) -> Tuple[POMOModel, torch.device]:
    import torch

    device_name = config["solver"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
        logger.warning("CUDA unavailable — falling back to CPU.")
    device = torch.device(device_name)

    model = POMOModel(
        node_feature_dim=config["pomo"]["node_feature_dim"],
        embedding_dim=config["pomo"]["embedding_dim"],
        num_heads=config["pomo"]["num_heads"],
        num_encoder_layers=config["pomo"]["num_encoder_layers"],
        ff_dim=config["pomo"]["feedforward_dim"],
    ).to(device)

    pretrained = config["training"].get("pretrained_model")
    if pretrained and os.path.isfile(pretrained):
        import torch
        checkpoint = torch.load(pretrained, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info("Loaded pretrained model from %s", pretrained)
    else:
        logger.info("No pretrained model — using random initialisation.")

    return model, device


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Branch & Price SD-VRPTW Solver")
    parser.add_argument("--config", default="configs/default_config.py")
    parser.add_argument("--run-name", default="bp_run")
    args = parser.parse_args()

    config = load_config(args.config)

    # First executable line: fix all seeds
    fix_all_seeds(config["solver"]["seed"])

    # Run-folder infrastructure
    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    run_folder = create_run_folder(results_dir, args.run_name)
    run_id = os.path.basename(run_folder)
    shadow_copy_config(args.config, run_folder)
    setup_logging(run_folder, config["logging"]["log_level"])

    logger.info("Run folder : %s", run_folder)
    logger.info("Config     : %s", args.config)

    wall_start = time.time()

    # Load problem
    problem = load_problem(
        orders_path=os.path.join(PROJECT_ROOT, config["problem"]["orders_file"]),
        trucks_path=os.path.join(PROJECT_ROOT, config["problem"]["trucks_file"]),
        distance_matrix_path=os.path.join(PROJECT_ROOT, config["problem"]["distance_matrix_file"]),
        depot_id=config["problem"]["depot_id"],
    )
    distance_matrix_path = (PROJECT_ROOT / config["problem"]["distance_matrix_file"]).resolve()
    geometry_path = distance_matrix_path.parent / f"geometry_{config['problem']['depot_id']}.json"
    if geometry_path.is_file():
        logger.info("Geometry file detected: %s", geometry_path)
    else:
        logger.info(
            "Geometry file not found next to distance matrix (expected: %s).",
            geometry_path,
        )
    logger.info(
        "Problem loaded: %d customers, %d vehicle types",
        problem.num_customers, len(problem.vehicle_types),
    )

    # Warm-start RMP
    column_pool = ColumnPool()
    build_initial_routes(problem, column_pool)
    print(column_pool)

    # Load POMO model
    model, device = load_pomo_model(config)
    
    # Initialize Pricing Orchestrator
    orchestrator = PricingOrchestrator(model, device, config)
    column_selector = build_column_selector(config, device)

    # Branch & Price
    solution = branch_and_price(
        problem,
        orchestrator,
        column_pool,
        config,
        column_selector=column_selector,
    )

    elapsed = time.time() - wall_start

    # ---- Report ----
    logger.info("\n" + "=" * 60)
    logger.info("SOLUTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cost          : %.4f", solution["total_cost"])
    logger.info("Number of routes    : %d", len(solution["routes"]))
    logger.info("B&B nodes explored  : %d", solution["nodes_explored"])
    logger.info("Total columns       : %d", solution["total_columns"])
    logger.info("Wall-clock time (s) : %.2f", elapsed)

    for route in solution["routes"]:
        cust_names = [problem.customer_ids[c] for c in route.customer_indices]
        logger.info(
            "  Route %d [%s] : %s  cost=%.2f",
            route.route_id, route.vehicle_type, cust_names, route.total_cost,
        )

    # ---- Persist ----
    raw = {
        "total_cost": solution["total_cost"],
        "num_routes": len(solution["routes"]),
        "nodes_explored": solution["nodes_explored"],
        "total_columns": solution["total_columns"],
        "cpu_time_seconds": elapsed,
        "distance_matrix_file": str(distance_matrix_path),
        "geometry_file": str(geometry_path) if geometry_path.is_file() else "",
        "forbidden_arcs": solution.get("forbidden_arcs", []),
        "enforced_arcs": solution.get("enforced_arcs", []),
        "routes": [
            {
                "route_id": r.route_id,
                "vehicle_type": r.vehicle_type,
                "customer_indices": r.customer_indices,
                "customer_ids": [problem.customer_ids[c] for c in r.customer_indices],
                "total_cost": r.total_cost,
                "distance_km": r.total_distance_km,
                "time_hours": r.total_time_hours,
            }
            for r in solution["routes"]
        ],
    }
    save_raw_output(raw, run_folder)

    append_to_master_csv(results_dir, run_id, {
        "total_cost": solution["total_cost"],
        "num_routes": len(solution["routes"]),
        "bb_nodes": solution["nodes_explored"],
        "total_columns": solution["total_columns"],
        "cpu_time_s": round(elapsed, 2),
    })

    logger.info("Results saved to %s", run_folder)


if __name__ == "__main__":
    main()
