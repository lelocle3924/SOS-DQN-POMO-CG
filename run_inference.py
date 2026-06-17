"""Root Node Column Generation Inference Script.
python run_inference.py --checkpoint results/YOUR_RUN/pretrained_pomo/final_model.pt --verbose

"""

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import fix_all_seeds, load_config
from src.data_loader import load_problem, ProblemData
from src.column_pool import ColumnPool
from src.master_problem import solve_master_problem
from src.pomo_model import POMOModel
from src.pricing_orchestrator import PricingOrchestrator

RUN_LOG_FILE_HANDLE = None


class TeeOutputStream:
    """Mirror terminal output to the terminal and a file."""

    def __init__(self, terminal_stream, log_file_stream):
        self.terminal_stream = terminal_stream
        self.log_file_stream = log_file_stream

    def write(self, message: str):
        self.terminal_stream.write(message)
        self.log_file_stream.write(message)
        return len(message)

    def flush(self):
        self.terminal_stream.flush()
        self.log_file_stream.flush()


def enable_run_log_capture(run_log_path: Path):
    """Capture everything printed to the terminal into run.log."""
    global RUN_LOG_FILE_HANDLE
    RUN_LOG_FILE_HANDLE = open(run_log_path, "w", encoding="utf-8")
    sys.stdout = TeeOutputStream(sys.stdout, RUN_LOG_FILE_HANDLE)
    sys.stderr = TeeOutputStream(sys.stderr, RUN_LOG_FILE_HANDLE)


def setup_inference_logging(verbose: bool):
    logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    # Remove existing handlers if any
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(message)s"  # Keep it clean for the trace
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def build_initial_routes(problem: ProblemData, column_pool: ColumnPool) -> int:
    """Hybrid greedy + single-customer fallback for RMP initialization."""
    global_unserved = set(range(problem.num_customers))

    # Primary Pass: Greedy Nearest-Neighbour
    for v_idx, vtype in enumerate(problem.vehicle_types):
        capacity = problem.vehicle_capacity[vtype]
        fixed_cost = problem.vehicle_fixed_cost[vtype]
        cost_per_km = problem.vehicle_cost_per_km[vtype]
        cost_per_hour = problem.vehicle_cost_per_hour[vtype]
        tt_mat = problem.travel_time_matrices[vtype]
        dist_km_mat = problem.distance_matrix_meters / 1000.0

        accessible = [
            c for c in range(problem.num_customers)
            if problem.site_dependency[c, v_idx]
        ]
        unserved_for_vtype = set(accessible)

        while unserved_for_vtype:
            route_customers = []
            cur_node = 0
            rem_cap = capacity
            cur_time = problem.depot_tw_start
            total_dist = 0.0

            while True:
                best, best_arr = None, float("inf")
                for c in unserved_for_vtype:
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
                total_dist += dist_km_mat[cur_node, node]
                arr = cur_time + tt_mat[cur_node, node]
                cur_time = max(arr, problem.tw_start[best]) + problem.service_times[best]
                rem_cap -= problem.demands[best]
                cur_node = node
                route_customers.append(best)
                unserved_for_vtype.discard(best)
                global_unserved.discard(best)

            if not route_customers:
                break

            total_dist += dist_km_mat[cur_node, 0]
            total_time = cur_time - problem.depot_tw_start + tt_mat[cur_node, 0]
            total_cost = fixed_cost + cost_per_km * total_dist + cost_per_hour * total_time

            column_pool.add_route(
                vehicle_type=vtype,
                customer_indices=route_customers,
                total_cost=total_cost,
                total_distance_km=total_dist,
                total_time_hours=total_time,
            )

    unserved_count = len(global_unserved)
    
    # Secondary Pass: Feasibility Fallback for unserved customers
    for c in global_unserved:
        allowed_v_idx = -1
        for v_idx in range(len(problem.vehicle_types)):
            if problem.site_dependency[c, v_idx]:
                allowed_v_idx = v_idx
                break
                
        if allowed_v_idx == -1:
            raise ValueError(f"Customer {c} has no allowed vehicle types!")
            
        vtype = problem.vehicle_types[allowed_v_idx]
        
        tt_mat = problem.travel_time_matrices[vtype]
        dk_mat = problem.distance_matrix_meters / 1000.0
        
        node = c + 1
        dist_km = dk_mat[0, node] + dk_mat[node, 0]
        
        arrival = problem.depot_tw_start + tt_mat[0, node]
        start_svc = max(arrival, problem.tw_start[c])
        cur_time = start_svc + problem.service_times[c]
        return_time = cur_time + tt_mat[node, 0]
        time_h = return_time - problem.depot_tw_start
        
        artificial_cost = 10000.0
        
        column_pool.add_route(
            vehicle_type=vtype,
            customer_indices=[c],
            total_cost=artificial_cost,
            total_distance_km=dist_km,
            total_time_hours=time_h,
        )
        
    return unserved_count


def resolve_configured_path(project_root: Path, configured_path: str) -> Path:
    raw_path = Path(configured_path)
    if raw_path.is_absolute():
        return raw_path
    return project_root / raw_path


def create_cg_results_folder(project_root: Path, orders_file_path: Path) -> Path:
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    folder_name = f"{orders_file_path.stem}_{timestamp}_cg"
    results_dir = project_root / "results" / folder_name
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def write_cg_markdown_report(
    results_dir: Path,
    orders_file_path: Path,
    problem: ProblemData,
    distance_matrix_path: Path,
    final_objective_value: float,
    total_iterations: int,
    total_columns: int,
    elapsed_seconds: float,
) -> Path:
    report_name = f"{results_dir.name}.md"
    report_path = results_dir / report_name
    report_lines = [
        "# Column Generation Inference Results",
        "",
        f"- Instance name: `{orders_file_path.name}`",
        f"- Number of customers: `{problem.num_customers}`",
        f"- Distance matrix: `{distance_matrix_path.name}`",
        f"- Final root LP objective: `{final_objective_value:.6f}`",
        f"- Total CG iterations: `{total_iterations}`",
        f"- Total columns in pool: `{total_columns}`",
        f"- Run time (seconds): `{elapsed_seconds:.6f}`",
    ]
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Root Node Inference with POMO")
    parser.add_argument("--config", default="configs/default_config.py", help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed DEBUG logging")
    parser.add_argument("--max-iterations", type=int, default=100, help="Override max CG iterations")
    parser.add_argument("--add-most-negative", action="store_true", help="Add only the single most negative reduced cost column per iteration")
    
    args = parser.parse_args()
    
    config_path = resolve_configured_path(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    orders_file_path = resolve_configured_path(PROJECT_ROOT, config["problem"]["orders_file"])
    trucks_file_path = resolve_configured_path(PROJECT_ROOT, config["problem"]["trucks_file"])
    distance_matrix_path = resolve_configured_path(PROJECT_ROOT, config["problem"]["distance_matrix_file"])
    results_dir = create_cg_results_folder(PROJECT_ROOT, orders_file_path)
    enable_run_log_capture(results_dir / "run.log")
    logger = setup_inference_logging(args.verbose)
    logger.info(f"Results folder created        : {results_dir}")

    fix_all_seeds(config["solver"]["seed"])

    logger.info("Loading problem data...")
    problem = load_problem(
        orders_path=str(orders_file_path),
        trucks_path=str(trucks_file_path),
        distance_matrix_path=str(distance_matrix_path),
        depot_id=config["problem"]["depot_id"],
    )
    
    device_name = config["solver"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading POMO model from {args.checkpoint}...")
    model = POMOModel(
        node_feature_dim=config["pomo"]["node_feature_dim"],
        embedding_dim=config["pomo"]["embedding_dim"],
        num_heads=config["pomo"]["num_heads"],
        num_encoder_layers=config["pomo"]["num_encoder_layers"],
        ff_dim=config["pomo"]["feedforward_dim"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    orchestrator = PricingOrchestrator(model, device, config)
    
    column_pool = ColumnPool()
    logger.info("Building initial basis with hybrid greedy heuristic...")
    unserved_count = build_initial_routes(problem, column_pool)
    logger.info(f"Greedy pass left {unserved_count} customers unserved, fallback applied.")
    
    logger.info("Starting Root Node Column Generation...\n")
    start_time = time.time()
    iteration = 0
    rmp = None
    
    max_iter = args.max_iterations
    rc_tol = config["column_generation"].get("reduced_cost_tolerance", -1e-5)
    
    # Suppress OR-Tools GLOP solver logs from polluting our clean trace
    logging.getLogger("ortools").setLevel(logging.WARNING)
    # Also suppress master_problem.py info logs since we log it here
    logging.getLogger("src.master_problem").setLevel(logging.WARNING)
    
    for iteration in range(1, max_iter + 1):
        rmp = solve_master_problem(column_pool, problem)
        
        if rmp.status != "OPTIMAL":
            logger.error(f"[Iteration {iteration}] RMP failed to solve! Status: {rmp.status}")
            break
            
        logger.info(f"[Iteration {iteration}] RMP Objective: {rmp.objective_value:.2f}")
        
        bb_constraints = {"forbidden_arcs": set(), "enforced_arcs": set()}
        new_routes = orchestrator.generate_columns(
            problem, rmp.dual_values, rmp.vehicle_dual_values, bb_constraints
        )
        
        if not new_routes:
            logger.info(f"[Iteration {iteration}] POMO generated 0 negative RC columns.")
            break
            
        # Sort ascending by reduced cost
        new_routes.sort(key=lambda r: r.reduced_cost)
        
        valid_negative_routes = [r for r in new_routes if r.reduced_cost < rc_tol]
        
        if not valid_negative_routes:
            logger.info(f"[Iteration {iteration}] POMO generated {len(new_routes)} columns, but none had RC < {rc_tol}.")
            break
            
        logger.info(f"[Iteration {iteration}] POMO generated {len(valid_negative_routes)} negative RC columns.")
        
        if args.add_most_negative:
            routes_to_add = [valid_negative_routes[0]]
        else:
            routes_to_add = valid_negative_routes
            
        logger.debug(f"[Iteration {iteration}] Adding Columns:")
        added_count = 0
        for r in routes_to_add:
            route_seq = [0] + [c + 1 for c in r.customer_indices] + [0]
            logger.debug(f"  - Vehicle: {r.vehicle_type} | RC: {r.reduced_cost:.2f} | Cost: {r.total_cost:.2f} | Route: {route_seq}")
            
            added = column_pool.add_route(
                vehicle_type=r.vehicle_type,
                customer_indices=r.customer_indices,
                total_cost=r.total_cost,
                total_distance_km=r.total_distance_km,
                total_time_hours=r.total_time_hours,
                reduced_cost=r.reduced_cost
            )
            if added is not None:
                added_count += 1
                
        if added_count == 0:
            logger.info(f"[Iteration {iteration}] No new unique columns added. Terminating.")
            break
            
        logger.info("") # Empty line for readability between iterations
            
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("ROOT NODE INFERENCE COMPLETE")
    logger.info("=" * 60)
    final_objective_value = float("nan")
    if rmp is not None and rmp.status == "OPTIMAL":
        final_objective_value = rmp.objective_value
    logger.info(f"Final Root Node LP Objective : {final_objective_value:.4f}")
    logger.info(f"Total CG Iterations          : {iteration}")
    logger.info(f"Total Columns in Pool        : {column_pool.num_routes}")
    logger.info(f"Total Execution Time         : {elapsed:.2f} seconds")

    shutil.copy2(config_path, results_dir / config_path.name)
    report_path = write_cg_markdown_report(
        results_dir=results_dir,
        orders_file_path=orders_file_path,
        problem=problem,
        distance_matrix_path=distance_matrix_path,
        final_objective_value=final_objective_value,
        total_iterations=iteration,
        total_columns=column_pool.num_routes,
        elapsed_seconds=elapsed,
    )
    logger.info(f"Results folder created        : {results_dir}")
    logger.info(f"Markdown report saved         : {report_path.name}")

if __name__ == "__main__":
    main()
