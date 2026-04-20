"""Run exact solver for the current config instance and write a report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ProblemData, load_problem
from src.exact_solver import ExactSolveResult, solve_exact_sdvrptw


@dataclass
class RunContext:
    instance_name: str
    customer_count: int
    distance_matrix_name: str
    distance_matrix_path: Path
    geometry_path: Path
    results_dir: Path


def _load_config(config_path: Path) -> Dict:
    with open(config_path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _resolve_path(project_root: Path, configured_path: str) -> Path:
    raw_path = Path(configured_path)
    if raw_path.is_absolute():
        return raw_path
    return project_root / raw_path


def _build_results_dir(base_results_dir: Path, orders_file_path: Path) -> Path:
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    folder_name = f"{orders_file_path.stem}_{timestamp}_exact"
    results_dir = base_results_dir / folder_name
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def _build_run_context(problem: ProblemData, distance_matrix_path: Path, results_dir: Path) -> RunContext:
    geometry_path = distance_matrix_path.parent / f"geometry_{problem.depot_id}.json"
    return RunContext(
        instance_name=problem.depot_id,
        customer_count=problem.num_customers,
        distance_matrix_name=distance_matrix_path.name,
        distance_matrix_path=distance_matrix_path.resolve(),
        geometry_path=geometry_path.resolve(),
        results_dir=results_dir,
    )


def _format_number(value: float) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    return f"{value:.6f}"


def _write_markdown_report(context: RunContext, exact_result: ExactSolveResult) -> None:
    report_path = context.results_dir / f"{context.results_dir.name}.md"
    report_lines = [
        "# Exact Solve Results",
        "",
        f"- Instance name: `{context.instance_name}`",
        f"- Number of customers: `{context.customer_count}`",
        f"- Distance matrix: `{context.distance_matrix_name}`",
        f"- Solver status: `{exact_result.status}`",
        f"- Objective value: `{_format_number(exact_result.objective_value)}`",
        f"- Best bound: `{_format_number(exact_result.best_bound)}`",
        f"- Run time (seconds): `{_format_number(exact_result.solve_time_seconds)}`",
        f"- Min gap percent: `{_format_number(exact_result.mip_gap_percent)}`",
        f"- Selected vehicle count: `{exact_result.selected_vehicle_count}`",
        f"- Number of routes: `{len(exact_result.routes)}`",
    ]
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")


def _write_raw_output(
    context: RunContext,
    problem: ProblemData,
    exact_result: ExactSolveResult,
) -> None:
    def _safe_float(value: float) -> Optional[float]:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)

    raw_payload = {
        "solver_status": exact_result.status,
        "total_cost": _safe_float(exact_result.objective_value),
        "num_routes": len(exact_result.routes),
        "nodes_explored": 0,
        "total_columns": 0,
        "cpu_time_seconds": float(exact_result.solve_time_seconds),
        "best_bound": _safe_float(exact_result.best_bound),
        "mip_gap_percent": _safe_float(exact_result.mip_gap_percent),
        "selected_vehicle_count": int(exact_result.selected_vehicle_count),
        "distance_matrix_file": str(context.distance_matrix_path),
        "geometry_file": str(context.geometry_path) if context.geometry_path.is_file() else "",
        "forbidden_arcs": [],
        "enforced_arcs": [],
        "routes": [
            {
                "route_id": route.route_id,
                "vehicle_type": route.vehicle_type,
                "customer_indices": route.customer_indices,
                "customer_ids": [problem.customer_ids[index] for index in route.customer_indices],
                "total_cost": route.total_cost,
                "distance_km": route.total_distance_km,
                "time_hours": route.total_time_hours,
            }
            for route in exact_result.routes
        ],
    }
    raw_output_path = context.results_dir / "raw_output.json"
    with open(raw_output_path, "w", encoding="utf-8") as raw_file:
        json.dump(raw_payload, raw_file, indent=2)


def _copy_config(config_path: Path, results_dir: Path) -> None:
    shutil.copy2(config_path, results_dir / "default_config.yaml")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact SD-VRPTW solver")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--time-limit-seconds",
        type=int,
        default=None,
        help="Override exact.time_limit_seconds from config",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    config_path = _resolve_path(PROJECT_ROOT, args.config)
    config = _load_config(config_path)
    exact_config = config.get("exact", {})
    configured_time_limit = int(exact_config.get("time_limit_seconds", 120))
    time_limit_seconds = args.time_limit_seconds if args.time_limit_seconds is not None else configured_time_limit
    assume_unlimited_vehicles = bool(exact_config.get("assume_unlimited_vehicles", True))
    max_vehicles_per_type = exact_config.get("max_vehicles_per_type", None)
    if max_vehicles_per_type is not None:
        max_vehicles_per_type = int(max_vehicles_per_type)

    orders_file_path = _resolve_path(PROJECT_ROOT, config["problem"]["orders_file"])
    trucks_file_path = _resolve_path(PROJECT_ROOT, config["problem"]["trucks_file"])
    distance_matrix_path = _resolve_path(PROJECT_ROOT, config["problem"]["distance_matrix_file"])
    depot_id = str(config["problem"]["depot_id"])
    configured_results_dir = config.get("logging", {}).get("results_dir", "results")
    base_results_dir = _resolve_path(PROJECT_ROOT, configured_results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)

    problem = load_problem(
        orders_path=str(orders_file_path),
        trucks_path=str(trucks_file_path),
        distance_matrix_path=str(distance_matrix_path),
        depot_id=depot_id,
    )

    results_dir = _build_results_dir(base_results_dir, orders_file_path)
    run_context = _build_run_context(problem, distance_matrix_path, results_dir)

    exact_result = solve_exact_sdvrptw(
        problem,
        time_limit_seconds=time_limit_seconds,
        assume_unlimited_vehicles=assume_unlimited_vehicles,
        max_vehicles_per_type=max_vehicles_per_type,
    )

    _write_markdown_report(run_context, exact_result)
    _write_raw_output(run_context, problem, exact_result)
    _copy_config(config_path, results_dir)

    print(f"Saved exact results to: {results_dir}")


if __name__ == "__main__":
    main()
