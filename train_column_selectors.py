"""Train pluggable selectors (RLCG / FFCG) from CG rollouts."""

from __future__ import annotations

import argparse
import csv
import logging
import json
import os
import re
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from run_inference import build_initial_routes
from src.column_pool import ColumnPool, Route
from src.column_selection.ffcg_selector import train_ffcg_selector
from src.column_selection.rlcg_env import RLCGEnvironment
from src.column_selection.rlcg_selector import train_rlcg_selector
from src.data_loader import (
    ProblemData,
    build_training_manifest,
    load_problem_for_training_instance,
)
from src.dp_solver import solve_espprc
from src.ga_generator import generate_columns_ga
from src.graph_dqn import (
    BipartiteGraphQNetwork,
    ColumnFeatureTracker,
    ReplayBuffer,
    ReplayTransition,
    build_bipartite_graph_state,
    clone_graph_state_to_cpu,
    select_candidate_indices,
)
from src.master_problem import solve_master_problem
from src.pomo_model import POMOModel
from src.pricing_orchestrator import PricingOrchestrator
from src.run_manager import setup_logging, shadow_copy_config
from src.utils import fix_all_seeds, load_config


PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


class SelectorTrainingTracker:
    """Optional TensorBoard tracker for selector training."""

    def __init__(self, enabled: bool, logdir: str) -> None:
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=logdir)
            logger.info("TensorBoard logging enabled at: %s", logdir)
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("TensorBoard unavailable (%s). Continuing without it.", exc)
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                full_key = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(full_key, float(value), step)

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def _load_mixed_training_problems(config: dict, max_instances: int) -> List[ProblemData]:
    training_cfg = config.get("training_data", {})
    temp_day_dirs = [
        str((PROJECT_ROOT / relative_path).resolve())
        for relative_path in training_cfg.get("temp_day_dirs", [])
    ]
    distance_matrix_dirs = [
        str((PROJECT_ROOT / relative_path).resolve())
        for relative_path in training_cfg.get("distance_matrix_dirs", [])
    ]
    manifest = build_training_manifest(
        temp_day_dirs=temp_day_dirs,
        distance_matrix_dirs=distance_matrix_dirs,
        distance_matrix_pattern=training_cfg.get(
            "distance_matrix_pattern", "distance_matrix_meters{depot_id}.csv"
        ),
    )
    selected_specs = manifest[:max_instances]
    trucks_path = str((PROJECT_ROOT / config["problem"]["trucks_file"]).resolve())

    problems: List[ProblemData] = []
    for instance_spec in selected_specs:
        problem = load_problem_for_training_instance(instance_spec, trucks_path=trucks_path)
        problems.append(problem)
    return problems


def _load_pomo_model(config: dict, device: torch.device) -> POMOModel:
    model = POMOModel(
        node_feature_dim=config["pomo"]["node_feature_dim"],
        embedding_dim=config["pomo"]["embedding_dim"],
        num_heads=config["pomo"]["num_heads"],
        num_encoder_layers=config["pomo"]["num_encoder_layers"],
        ff_dim=config["pomo"]["feedforward_dim"],
    ).to(device)
    pretrained_model = config.get("training", {}).get("pretrained_model")
    if pretrained_model:
        checkpoint = torch.load(pretrained_model, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _generate_candidates_from_dp(
    problem: ProblemData,
    dual_values: np.ndarray,
    vehicle_duals: dict,
    config: dict,
) -> List[Route]:
    rc_tol = float(config["column_generation"].get("reduced_cost_tolerance", -1e-6))
    dp_cfg = config.get("dp_solver", {})
    beam_width = int(dp_cfg.get("beam_width", 50))
    max_labels = int(dp_cfg.get("max_total_labels", 10000))
    candidates: List[Route] = []

    for vehicle_index, vehicle_type in enumerate(problem.vehicle_types):
        dp_results = solve_espprc(
            problem=problem,
            dual_values=dual_values,
            vehicle_type=vehicle_type,
            vehicle_idx=vehicle_index,
            forbidden_arcs=set(),
            enforced_arcs=set(),
            vehicle_dual=float(vehicle_duals.get(vehicle_type, 0.0)),
            beam_width=beam_width,
            max_total_labels=max_labels,
        )
        for customer_indices, total_cost, distance_km, time_hours, reduced_cost in dp_results:
            if reduced_cost < rc_tol:
                candidates.append(
                    Route(
                        route_id=-1,
                        vehicle_type=vehicle_type,
                        customer_indices=customer_indices,
                        visit_sequence=customer_indices,
                        total_cost=total_cost,
                        total_distance_km=distance_km,
                        total_time_hours=time_hours,
                        reduced_cost=reduced_cost,
                    )
                )
    return candidates


def _generate_candidates_from_ga(
    problem: ProblemData,
    dual_values: np.ndarray,
    vehicle_duals: dict,
    config: dict,
) -> List[Route]:
    rc_tol = float(config["column_generation"].get("reduced_cost_tolerance", -1e-6))
    ga_cfg = config.get("ga_generator", {})
    population_size = int(ga_cfg.get("population_size", 50))
    num_generations = int(ga_cfg.get("num_generations", 100))
    mutation_rate = float(ga_cfg.get("mutation_rate", 0.3))
    max_route_length = int(ga_cfg.get("max_route_length", 20))

    candidates: List[Route] = []
    for vehicle_index, vehicle_type in enumerate(problem.vehicle_types):
        ga_results = generate_columns_ga(
            problem=problem,
            dual_values=dual_values,
            vehicle_type=vehicle_type,
            vehicle_idx=vehicle_index,
            vehicle_dual=float(vehicle_duals.get(vehicle_type, 0.0)),
            population_size=population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            max_route_length=max_route_length,
        )
        for customer_indices, total_cost, distance_km, time_hours, reduced_cost in ga_results:
            if reduced_cost < rc_tol:
                candidates.append(
                    Route(
                        route_id=-1,
                        vehicle_type=vehicle_type,
                        customer_indices=customer_indices,
                        visit_sequence=customer_indices,
                        total_cost=total_cost,
                        total_distance_km=distance_km,
                        total_time_hours=time_hours,
                        reduced_cost=reduced_cost,
                    )
                )
    return candidates


def _generate_candidates(
    source: str,
    orchestrator: PricingOrchestrator,
    problem: ProblemData,
    dual_values: np.ndarray,
    vehicle_duals: dict,
    config: dict,
) -> List[Route]:
    if source == "pomo":
        return orchestrator.generate_columns(
            problem=problem,
            dual_values=dual_values,
            vehicle_dual_values=vehicle_duals,
            bb_constraints={"forbidden_arcs": set(), "enforced_arcs": set()},
        )
    if source == "ga":
        return _generate_candidates_from_ga(problem, dual_values, vehicle_duals, config)
    if source == "dp":
        return _generate_candidates_from_dp(problem, dual_values, vehicle_duals, config)
    raise ValueError(f"Unsupported candidate source: {source}")


def _collect_rlcg_samples(
    problems: Sequence[ProblemData],
    orchestrator: PricingOrchestrator,
    config: dict,
    candidate_source: str,
    max_iterations: int,
    tb_tracker: Optional[SelectorTrainingTracker] = None,
) -> List[Tuple]:
    samples: List[Tuple] = []
    feature_tracker = ColumnFeatureTracker()
    total_problems = max(1, len(problems))
    collect_start_time = time.time()

    for problem_index, problem in enumerate(problems, start=1):
        column_pool = ColumnPool()
        build_initial_routes(problem, column_pool)
        for _ in range(max_iterations):
            rmp_result = solve_master_problem(column_pool, problem)
            if rmp_result.status != "OPTIMAL":
                break
            feature_tracker.update_from_rmp(column_pool.routes, rmp_result.route_weights)
            candidates = _generate_candidates(
                source=candidate_source,
                orchestrator=orchestrator,
                problem=problem,
                dual_values=rmp_result.dual_values,
                vehicle_duals=rmp_result.vehicle_dual_values,
                config=config,
            )
            if not candidates:
                break
            state = build_bipartite_graph_state(
                column_pool=column_pool,
                rmp_result=rmp_result,
                candidate_routes=candidates,
                num_customers=problem.num_customers,
                tracker=feature_tracker,
            )
            best_index = int(
                np.argmin(
                    [route.reduced_cost if route.reduced_cost is not None else 1e18 for route in candidates]
                )
            )
            samples.append((state, best_index))
            chosen_route = candidates[best_index]
            column_pool.add_route(
                vehicle_type=chosen_route.vehicle_type,
                customer_indices=chosen_route.customer_indices,
                total_cost=chosen_route.total_cost,
                total_distance_km=chosen_route.total_distance_km,
                total_time_hours=chosen_route.total_time_hours,
                reduced_cost=chosen_route.reduced_cost,
            )
        progress = problem_index / total_problems
        elapsed_seconds = time.time() - collect_start_time
        eta_seconds = (elapsed_seconds / progress) - elapsed_seconds if progress > 0 else float("nan")
        logger.info(
            "RLCG sample collection: %d/%d problems (%.1f%%)  samples=%d  eta=%.1fs",
            problem_index,
            total_problems,
            progress * 100.0,
            len(samples),
            eta_seconds,
        )
        if tb_tracker is not None:
            tb_tracker.log_scalar("collection/rlcg_progress_percent", progress * 100.0, problem_index)
            tb_tracker.log_scalar("collection/rlcg_samples", float(len(samples)), problem_index)
            tb_tracker.log_scalar("collection/rlcg_eta_seconds", float(eta_seconds), problem_index)
    return samples


def _collect_ffcg_samples(
    problems: Sequence[ProblemData],
    orchestrator: PricingOrchestrator,
    config: dict,
    candidate_source: str,
    max_iterations: int,
    max_family_size: int,
    tb_tracker: Optional[SelectorTrainingTracker] = None,
) -> List[Tuple]:
    samples: List[Tuple] = []
    feature_tracker = ColumnFeatureTracker()
    rc_tol = float(config["column_generation"].get("reduced_cost_tolerance", -1e-6))
    total_problems = max(1, len(problems))
    collect_start_time = time.time()

    for problem_index, problem in enumerate(problems, start=1):
        column_pool = ColumnPool()
        build_initial_routes(problem, column_pool)
        for _ in range(max_iterations):
            rmp_result = solve_master_problem(column_pool, problem)
            if rmp_result.status != "OPTIMAL":
                break
            feature_tracker.update_from_rmp(column_pool.routes, rmp_result.route_weights)
            candidates = _generate_candidates(
                source=candidate_source,
                orchestrator=orchestrator,
                problem=problem,
                dual_values=rmp_result.dual_values,
                vehicle_duals=rmp_result.vehicle_dual_values,
                config=config,
            )
            if not candidates:
                break

            remaining_candidates = list(candidates)
            selected_count = 0
            while remaining_candidates and selected_count < max_family_size:
                state = build_bipartite_graph_state(
                    column_pool=column_pool,
                    rmp_result=rmp_result,
                    candidate_routes=remaining_candidates,
                    num_customers=problem.num_customers,
                    tracker=feature_tracker,
                )
                teacher_index = int(
                    np.argmin(
                        [
                            route.reduced_cost if route.reduced_cost is not None else 1e18
                            for route in remaining_candidates
                        ]
                    )
                )
                teacher_route = remaining_candidates[teacher_index]
                if teacher_route.reduced_cost is None or teacher_route.reduced_cost >= rc_tol:
                    break

                samples.append((state, teacher_index))
                column_pool.add_route(
                    vehicle_type=teacher_route.vehicle_type,
                    customer_indices=teacher_route.customer_indices,
                    total_cost=teacher_route.total_cost,
                    total_distance_km=teacher_route.total_distance_km,
                    total_time_hours=teacher_route.total_time_hours,
                    reduced_cost=teacher_route.reduced_cost,
                )
                remaining_candidates.pop(teacher_index)
                selected_count += 1
        progress = problem_index / total_problems
        elapsed_seconds = time.time() - collect_start_time
        eta_seconds = (elapsed_seconds / progress) - elapsed_seconds if progress > 0 else float("nan")
        logger.info(
            "FFCG sample collection: %d/%d problems (%.1f%%)  samples=%d  eta=%.1fs",
            problem_index,
            total_problems,
            progress * 100.0,
            len(samples),
            eta_seconds,
        )
        if tb_tracker is not None:
            tb_tracker.log_scalar("collection/ffcg_progress_percent", progress * 100.0, problem_index)
            tb_tracker.log_scalar("collection/ffcg_samples", float(len(samples)), problem_index)
            tb_tracker.log_scalar("collection/ffcg_eta_seconds", float(eta_seconds), problem_index)
    return samples


def _resolve_output_checkpoint_path(
    output_argument: str,
    default_checkpoint_dir: str,
    default_checkpoint_name: str,
) -> Path:
    """Build a cross-platform checkpoint path from CLI input."""
    if not output_argument.strip():
        return (Path(default_checkpoint_dir) / default_checkpoint_name).resolve()

    normalized_output = output_argument.strip()
    if os.name != "nt":
        normalized_output = normalized_output.replace("\\", "/")

    requested_path = Path(normalized_output).expanduser()
    if requested_path.is_absolute():
        return requested_path
    return (PROJECT_ROOT / requested_path).resolve()


def _extract_epoch_from_resume_checkpoint(
    resume_checkpoint_path: str,
    loaded_checkpoint: Dict[str, Any],
) -> int:
    """Resolve last completed epoch from checkpoint metadata or filename."""
    raw_epoch_value = loaded_checkpoint.get("epoch")
    if isinstance(raw_epoch_value, (int, float)):
        return max(0, int(raw_epoch_value))

    filename_match = re.search(r"_epoch_(\d+)\.pt$", Path(resume_checkpoint_path).name)
    if filename_match is None:
        return 0
    return int(filename_match.group(1))


def _build_selector_checkpoint_name(method: str, epoch_number: int) -> str:
    """Return standardized selector checkpoint filename."""
    return f"dqn_model_{method}_epoch_{epoch_number}.pt"


def _find_latest_selector_checkpoint(checkpoint_dir: str, method: str) -> Optional[Path]:
    """Find latest epoch checkpoint for a selector method."""
    checkpoint_directory_path = Path(checkpoint_dir)
    if not checkpoint_directory_path.exists():
        return None

    matching_checkpoints = sorted(
        checkpoint_directory_path.glob(f"dqn_model_{method}_epoch_*.pt")
    )
    if not matching_checkpoints:
        return None

    def _extract_epoch_from_filename(checkpoint_path: Path) -> int:
        match = re.search(r"_epoch_(\d+)\.pt$", checkpoint_path.name)
        return int(match.group(1)) if match is not None else -1

    return max(matching_checkpoints, key=_extract_epoch_from_filename)


def _initialize_epoch_metrics_csv(
    metrics_csv_path: str,
    resume_base_epoch: int,
) -> None:
    """Create per-epoch metrics file or trim to resume epoch."""
    csv_header = [
        "epoch",
        "loss",
        "td_loss",
        "avg_reward",
        "episode_length",
        "epsilon",
        "q_value_mean",
        "replay_size",
        "target_sync_count",
        "elapsed_seconds",
        "eta_seconds",
        "progress_percent",
        "num_samples",
        "method",
        "candidate_source",
        "training_mode",
    ]
    metrics_path = Path(metrics_csv_path)
    if resume_base_epoch <= 0 or not metrics_path.exists():
        with open(metrics_path, "w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.writer(csv_handle)
            writer.writerow(csv_header)
        return

    with open(metrics_path, "r", encoding="utf-8") as csv_handle:
        existing_rows = list(csv.reader(csv_handle))
    if not existing_rows:
        rows_to_keep = [csv_header]
    else:
        rows_to_keep = [existing_rows[0]]
        for row in existing_rows[1:]:
            if not row:
                continue
            if row[0].isdigit() and int(row[0]) <= resume_base_epoch:
                rows_to_keep.append(row)
    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerows(rows_to_keep)


def _append_epoch_metrics_csv_row(
    metrics_csv_path: str,
    absolute_epoch_index: int,
    train_stats: Dict[str, float],
    num_samples: int,
    method: str,
    candidate_source: str,
    training_mode: str,
) -> None:
    """Append a single epoch training row for plotting."""
    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(
            [
                absolute_epoch_index,
                float(train_stats.get("loss", float("nan"))),
                float(train_stats.get("td_loss", float("nan"))),
                float(train_stats.get("avg_reward", float("nan"))),
                float(train_stats.get("episode_length", float("nan"))),
                float(train_stats.get("epsilon", float("nan"))),
                float(train_stats.get("q_value_mean", float("nan"))),
                float(train_stats.get("replay_size", float("nan"))),
                float(train_stats.get("target_sync_count", float("nan"))),
                float(train_stats.get("elapsed_seconds", float("nan"))),
                float(train_stats.get("eta_seconds", float("nan"))),
                float(train_stats.get("progress_percent", float("nan"))),
                num_samples,
                method,
                candidate_source,
                training_mode,
            ]
        )


def _parse_override_value(raw_value: str) -> Any:
    """Parse CLI override value via YAML scalars/lists/dicts."""
    return yaml.safe_load(raw_value)


def _set_nested_config_value(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set config using dot-path notation, creating nested dictionaries as needed."""
    key_parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not key_parts:
        raise ValueError(f"Invalid override key: '{dotted_key}'")
    current_level: Dict[str, Any] = config
    for key_part in key_parts[:-1]:
        existing = current_level.get(key_part)
        if existing is None:
            current_level[key_part] = {}
            existing = current_level[key_part]
        if not isinstance(existing, dict):
            raise ValueError(
                f"Cannot set nested override '{dotted_key}': '{key_part}' is not a dictionary."
            )
        current_level = existing
    current_level[key_parts[-1]] = value


def _apply_config_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply repeated --set key=value overrides."""
    for override_entry in overrides:
        if "=" not in override_entry:
            raise ValueError(
                f"Invalid --set value '{override_entry}'. Expected format: key.path=value"
            )
        raw_key, raw_value = override_entry.split("=", 1)
        key = raw_key.strip()
        value = _parse_override_value(raw_value.strip())
        _set_nested_config_value(config, key, value)
        logger.info("Applied config override: %s=%r", key, value)


def _resolve_rlcg_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve RLCG DQN hyperparameters with safe defaults."""
    training_cfg = dict(config.get("rlcg_training", {}))
    resolved = {
        "alpha": float(training_cfg.get("alpha", 1.0)),
        "gamma": float(training_cfg.get("gamma", 0.99)),
        "epsilon_start": float(training_cfg.get("epsilon_start", 1.0)),
        "epsilon_end": float(training_cfg.get("epsilon_end", 0.05)),
        "epsilon_decay_steps": int(training_cfg.get("epsilon_decay_steps", 1000)),
        "replay_capacity": int(training_cfg.get("replay_capacity", 5000)),
        "min_replay_size": int(training_cfg.get("min_replay_size", 128)),
        "batch_size": int(training_cfg.get("batch_size", 32)),
        "train_steps_per_collect": int(training_cfg.get("train_steps_per_collect", 10)),
        "target_update_interval": int(training_cfg.get("target_update_interval", 50)),
        "max_episode_steps": int(training_cfg.get("max_episode_steps", 8)),
        "gradient_clip_norm": float(training_cfg.get("gradient_clip_norm", 1.0)),
    }
    if resolved["epsilon_decay_steps"] <= 0:
        raise ValueError("rlcg_training.epsilon_decay_steps must be positive.")
    if resolved["replay_capacity"] <= 0:
        raise ValueError("rlcg_training.replay_capacity must be positive.")
    if resolved["batch_size"] <= 0:
        raise ValueError("rlcg_training.batch_size must be positive.")
    return resolved


def _epsilon_by_step(total_steps: int, cfg: Dict[str, Any]) -> float:
    decay_progress = min(1.0, float(total_steps) / float(cfg["epsilon_decay_steps"]))
    return float(cfg["epsilon_start"] + (cfg["epsilon_end"] - cfg["epsilon_start"]) * decay_progress)


def _max_masked_q_value(
    q_values: torch.Tensor,
    action_nodes: torch.Tensor,
    action_mask: Sequence[bool],
) -> Optional[torch.Tensor]:
    if action_nodes.numel() == 0:
        return None
    if len(action_mask) != int(action_nodes.numel()):
        return None
    mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device)
    if not bool(mask_tensor.any()):
        return None
    action_q = q_values[action_nodes]
    masked_q = torch.full_like(action_q, fill_value=-float("inf"))
    masked_q[mask_tensor] = action_q[mask_tensor]
    return torch.max(masked_q)


def _train_dqn_from_replay(
    q_online: BipartiteGraphQNetwork,
    q_target: BipartiteGraphQNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    gamma: float,
    gradient_clip_norm: float,
) -> Dict[str, float]:
    transitions = replay_buffer.sample(batch_size)
    if not transitions:
        return {"td_loss": 0.0, "q_value_mean": 0.0}

    losses: List[torch.Tensor] = []
    q_values_logged: List[float] = []
    for transition in transitions:
        state = transition.state.to(device)
        q_all = q_online(state)
        num_actions = int(state.action_node_indices.numel())
        if transition.action_local_index < 0 or transition.action_local_index >= num_actions:
            continue
        action_node = state.action_node_indices[transition.action_local_index]
        predicted_q = q_all[action_node]
        q_values_logged.append(float(predicted_q.detach().item()))

        with torch.no_grad():
            if transition.done or transition.next_state is None:
                target_q_value = torch.tensor(float(transition.reward), dtype=torch.float32, device=device)
            else:
                next_state = transition.next_state.to(device)
                next_q_target_all = q_target(next_state)
                best_next_q = _max_masked_q_value(
                    q_values=next_q_target_all,
                    action_nodes=next_state.action_node_indices,
                    action_mask=transition.next_action_mask or [],
                )
                bootstrap_value = 0.0 if best_next_q is None else float(best_next_q.item())
                target_q_value = torch.tensor(
                    float(transition.reward) + float(gamma) * bootstrap_value,
                    dtype=torch.float32,
                    device=device,
                )

        losses.append(torch.nn.functional.mse_loss(predicted_q, target_q_value))

    if not losses:
        return {"td_loss": 0.0, "q_value_mean": 0.0}
    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    if gradient_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(q_online.parameters(), max_norm=gradient_clip_norm)
    optimizer.step()
    return {
        "td_loss": float(total_loss.item()),
        "q_value_mean": float(np.mean(q_values_logged)) if q_values_logged else 0.0,
    }


def _train_rlcg_dqn(
    q_model: BipartiteGraphQNetwork,
    problems: Sequence[ProblemData],
    orchestrator: PricingOrchestrator,
    config: dict,
    candidate_source: str,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    rl_cfg = _resolve_rlcg_training_config(config)
    rc_tol = float(config["column_generation"].get("reduced_cost_tolerance", -1e-6))
    q_model.to(device)
    q_model.train()
    q_target = BipartiteGraphQNetwork(column_feature_dim=9, constraint_feature_dim=2, hidden_dim=q_model.column_encoder.out_features).to(device)
    q_target.load_state_dict(q_model.state_dict())
    q_target.eval()

    replay_buffer = ReplayBuffer(capacity=rl_cfg["replay_capacity"])
    optimizer = torch.optim.Adam(q_model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed=int(config["solver"]["seed"]))
    total_collect_steps = 0
    target_sync_count = 0
    gradient_step_count = 0
    total_episodes = 0
    epoch_td_losses: List[float] = []
    epoch_rewards: List[float] = []
    epoch_lengths: List[float] = []
    train_start_time = time.time()

    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        for problem in problems:
            env = RLCGEnvironment(
                problem=problem,
                candidate_generator=lambda env_problem, dual_values, vehicle_duals: _generate_candidates(
                    source=candidate_source,
                    orchestrator=orchestrator,
                    problem=env_problem,
                    dual_values=dual_values,
                    vehicle_duals=vehicle_duals,
                    config=config,
                ),
                build_initial_pool=build_initial_routes,
                reduced_cost_tolerance=rc_tol,
                alpha=rl_cfg["alpha"],
                max_episode_steps=rl_cfg["max_episode_steps"],
            )
            observation = env.reset()
            done = bool(observation.done)
            cumulative_reward = 0.0
            episode_step_count = 0

            while not done:
                if observation.state is None:
                    break
                epsilon = _epsilon_by_step(total_collect_steps, rl_cfg)
                selected_indices = select_candidate_indices(
                    model=q_model,
                    state=observation.state,
                    max_selected=1,
                    action_mask=observation.action_mask,
                    epsilon=epsilon,
                    rng=rng,
                    device=device,
                )
                if not selected_indices:
                    done = True
                    break

                action_index = int(selected_indices[0])
                next_observation, reward = env.step(action_index)
                transition = ReplayTransition(
                    state=clone_graph_state_to_cpu(observation.state),
                    action_local_index=action_index,
                    reward=float(reward),
                    next_state=clone_graph_state_to_cpu(next_observation.state)
                    if next_observation.state is not None
                    else None,
                    done=bool(next_observation.done),
                    next_action_mask=list(next_observation.action_mask),
                )
                
                # Burn-In Phase Logic: Only add to replay buffer if no dummy variables are in basis
                dummy_in_basis = False
                if env._current_rmp is not None and env._column_pool is not None:
                    for idx, weight in enumerate(env._current_rmp.route_weights):
                        if weight > 1e-6 and env._column_pool.routes[idx].total_cost >= 9000.0:
                            dummy_in_basis = True
                            break
                            
                if not dummy_in_basis:
                    replay_buffer.add(transition)
                    
                total_collect_steps += 1
                cumulative_reward += float(reward)
                episode_step_count += 1
                observation = next_observation
                done = bool(next_observation.done)

            total_episodes += 1
            episode_rewards.append(cumulative_reward)
            episode_lengths.append(episode_step_count)

        td_losses_this_epoch: List[float] = []
        q_values_this_epoch: List[float] = []
        if len(replay_buffer) >= rl_cfg["min_replay_size"]:
            for _ in range(rl_cfg["train_steps_per_collect"]):
                train_stats = _train_dqn_from_replay(
                    q_online=q_model,
                    q_target=q_target,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    device=device,
                    batch_size=rl_cfg["batch_size"],
                    gamma=rl_cfg["gamma"],
                    gradient_clip_norm=rl_cfg["gradient_clip_norm"],
                )
                td_losses_this_epoch.append(float(train_stats["td_loss"]))
                q_values_this_epoch.append(float(train_stats["q_value_mean"]))
                gradient_step_count += 1
                if gradient_step_count % rl_cfg["target_update_interval"] == 0:
                    q_target.load_state_dict(q_model.state_dict())
                    target_sync_count += 1

        epoch_td_loss = float(np.mean(td_losses_this_epoch)) if td_losses_this_epoch else 0.0
        epoch_avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        epoch_avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        epoch_avg_q = float(np.mean(q_values_this_epoch)) if q_values_this_epoch else 0.0
        epoch_td_losses.append(epoch_td_loss)
        epoch_rewards.append(epoch_avg_reward)
        epoch_lengths.append(epoch_avg_length)
        epoch_elapsed_seconds = time.time() - epoch_start_time
        elapsed_seconds = time.time() - train_start_time
        progress_ratio = float(epoch_index + 1) / max(1, num_epochs)
        eta_seconds = (elapsed_seconds / progress_ratio - elapsed_seconds) if progress_ratio > 0.0 else float("nan")
        logger.info(
            "RLCG-DQN epoch %d/%d loss=%.6f reward=%.4f eps=%.4f replay=%d epoch_time=%.1fs",
            epoch_index + 1,
            num_epochs,
            epoch_td_loss,
            epoch_avg_reward,
            _epsilon_by_step(total_collect_steps, rl_cfg),
            len(replay_buffer),
            epoch_elapsed_seconds,
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": float(epoch_index + 1),
                    "num_epochs": float(num_epochs),
                    "progress_percent": float(progress_ratio * 100.0),
                    "loss": epoch_td_loss,
                    "td_loss": epoch_td_loss,
                    "avg_reward": epoch_avg_reward,
                    "episode_length": epoch_avg_length,
                    "epsilon": _epsilon_by_step(total_collect_steps, rl_cfg),
                    "q_value_mean": epoch_avg_q,
                    "replay_size": float(len(replay_buffer)),
                    "target_sync_count": float(target_sync_count),
                    "num_samples": float(len(replay_buffer)),
                    "elapsed_seconds": float(elapsed_seconds),
                    "eta_seconds": float(eta_seconds),
                }
            )
    return {
        "loss": float(np.mean(epoch_td_losses)) if epoch_td_losses else 0.0,
        "td_loss": float(np.mean(epoch_td_losses)) if epoch_td_losses else 0.0,
        "avg_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
        "episode_length": float(np.mean(epoch_lengths)) if epoch_lengths else 0.0,
        "replay_size": float(len(replay_buffer)),
        "target_sync_count": float(target_sync_count),
        "num_episodes": float(total_episodes),
        "num_samples": float(len(replay_buffer)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RLCG or FFCG selector.")
    parser.add_argument("--config", default="configs/default_config.py")
    parser.add_argument("--method", choices=["rlcg", "ffcg"], required=True)
    parser.add_argument(
        "--training-mode",
        choices=["dqn", "imitation"],
        default="dqn",
        help="RLCG mode: dqn (paper-style) or imitation baseline.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional suffix for run folder name. Default: col_selector_<method>",
    )
    parser.add_argument("--results-dir", default="", help="Override config logging.results_dir")
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--candidate-source", choices=["pomo", "ga", "dp"], default="pomo")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save selector checkpoint every N training epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional selector checkpoint (.pt) to continue training from.",
    )
    parser.add_argument(
        "--resume-dir",
        default="",
        help="Resume in an existing run folder and continue checkpoint/metrics there.",
    )
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument(
        "--tb-logdir",
        default="",
        help="TensorBoard log directory. Defaults to results/tensorboard_selectors/<method>_<source>_<timestamp>",
    )
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with key.path=value. Repeatable.",
    )
    args = parser.parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be a positive integer.")
    if args.resume_dir.strip() and args.resume_from.strip():
        raise ValueError("Use either --resume-dir or --resume-from, not both.")
    effective_training_mode = args.training_mode
    if args.method == "ffcg" and args.training_mode != "imitation":
        effective_training_mode = "imitation"

    config_path = str((PROJECT_ROOT / args.config).resolve())
    config = load_config(config_path)
    _apply_config_overrides(config, args.set)
    if args.results_dir.strip():
        config["logging"]["results_dir"] = args.results_dir.strip()

    results_dir = str((PROJECT_ROOT / config["logging"]["results_dir"]).resolve())
    os.makedirs(results_dir, exist_ok=True)
    resume_mode = False
    if args.resume_dir.strip():
        run_folder = str(Path(args.resume_dir.strip()).expanduser().resolve())
        if not os.path.isdir(run_folder):
            raise FileNotFoundError(f"--resume-dir does not exist: {run_folder}")
        resume_mode = True
        setup_logging(run_folder, config["logging"].get("log_level", "INFO"))
        logger.info("Resuming in existing run folder: %s", run_folder)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_suffix = args.run_name.strip() or f"col_selector_{args.method}"
        run_folder = os.path.join(results_dir, f"train_{timestamp}_{run_suffix}")
        os.makedirs(run_folder, exist_ok=True)
        shadow_copy_config(config_path, run_folder)
        setup_logging(run_folder, config["logging"].get("log_level", "INFO"))
        logger.info("Run folder: %s", run_folder)
    if args.method == "ffcg" and args.training_mode != "imitation":
        logger.warning("FFCG currently supports imitation mode only. Using training_mode=imitation.")

    fix_all_seeds(config["solver"]["seed"])
    device = torch.device("cpu")
    default_tb_logdir = str(
        (Path(run_folder) / "tensorboard").resolve()
    )
    tb_logdir = args.tb_logdir.strip() or default_tb_logdir
    tracker = SelectorTrainingTracker(enabled=args.tensorboard, logdir=tb_logdir)
    ckpt_dir = os.path.join(run_folder, "pretrained_col_selector")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_csv_path = os.path.join(run_folder, "training_metrics.csv")

    try:
        problems = _load_mixed_training_problems(config, max_instances=args.max_instances)
        pomo_model = _load_pomo_model(config, device)
        orchestrator = PricingOrchestrator(pomo_model, device, config)

        q_model = BipartiteGraphQNetwork(
            column_feature_dim=9,
            constraint_feature_dim=2,
            hidden_dim=int(config.get("column_selector", {}).get("hidden_dim", 64)),
        )
        resume_base_epoch = 0
        resume_checkpoint_path = ""
        if resume_mode:
            latest_resume_checkpoint = _find_latest_selector_checkpoint(ckpt_dir, args.method)
            if latest_resume_checkpoint is None:
                raise FileNotFoundError(
                    f"No checkpoint found in {ckpt_dir} matching method '{args.method}'."
                )
            resume_checkpoint_path = str(latest_resume_checkpoint)
        elif args.resume_from.strip():
            resume_checkpoint_path = args.resume_from.strip()

        if resume_checkpoint_path:
            resume_path = resume_checkpoint_path
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"--resume-from checkpoint not found: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            q_model.load_state_dict(state_dict)
            if isinstance(checkpoint, dict):
                resume_base_epoch = _extract_epoch_from_resume_checkpoint(
                    resume_checkpoint_path=resume_path,
                    loaded_checkpoint=checkpoint,
                )
            else:
                resume_base_epoch = _extract_epoch_from_resume_checkpoint(
                    resume_checkpoint_path=resume_path,
                    loaded_checkpoint={},
                )
            logger.info("Loaded selector weights from: %s", resume_path)
            logger.info("Resuming selector epoch numbering from: %d", resume_base_epoch + 1)

        _initialize_epoch_metrics_csv(metrics_csv_path, resume_base_epoch)
        latest_checkpoint_path: Optional[Path] = None
        latest_checkpoint_epoch = 0
        last_logged_epoch = resume_base_epoch
        common_checkpoint_payload: Dict[str, Any] = {
            "method": args.method,
            "training_mode": effective_training_mode,
            "candidate_source": args.candidate_source,
            "run_folder": run_folder,
            "resume_from": resume_checkpoint_path or None,
        }

        def _save_epoch_checkpoint(epoch_number: int, metrics_payload: Dict[str, float]) -> Path:
            checkpoint_path = Path(ckpt_dir) / _build_selector_checkpoint_name(args.method, epoch_number)
            checkpoint_data = {
                "model_state_dict": q_model.state_dict(),
                "epoch": int(epoch_number),
                "train_metrics": dict(metrics_payload),
                "num_samples": int(sample_count_for_logs),
                **common_checkpoint_payload,
            }
            torch.save(checkpoint_data, str(checkpoint_path))
            return checkpoint_path

        train_start_time = time.time()
        sample_count_for_logs = 0
        if args.method == "rlcg":
            samples: List[Tuple] = []
            if effective_training_mode == "imitation":
                samples = _collect_rlcg_samples(
                    problems=problems,
                    orchestrator=orchestrator,
                    config=config,
                    candidate_source=args.candidate_source,
                    max_iterations=args.max_iterations,
                    tb_tracker=tracker,
                )
                sample_count_for_logs = len(samples)

            def _on_train_progress(train_stats: Dict[str, float]) -> None:
                nonlocal latest_checkpoint_path, latest_checkpoint_epoch, last_logged_epoch, sample_count_for_logs
                local_epoch_index = int(train_stats.get("epoch", 0))
                absolute_epoch_index = resume_base_epoch + local_epoch_index
                last_logged_epoch = absolute_epoch_index
                sample_count_for_logs = int(train_stats.get("num_samples", sample_count_for_logs))
                _append_epoch_metrics_csv_row(
                    metrics_csv_path=metrics_csv_path,
                    absolute_epoch_index=absolute_epoch_index,
                    train_stats=train_stats,
                    num_samples=len(samples),
                    method=args.method,
                    candidate_source=args.candidate_source,
                    training_mode=effective_training_mode,
                )
                tracker.log_metrics(train_stats, step=absolute_epoch_index, prefix="train/rlcg")
                if absolute_epoch_index % args.checkpoint_interval == 0:
                    latest_checkpoint_path = _save_epoch_checkpoint(absolute_epoch_index, train_stats)
                    latest_checkpoint_epoch = absolute_epoch_index
                    logger.info("Saved selector checkpoint: %s", latest_checkpoint_path.name)

            if effective_training_mode == "imitation":
                metrics = train_rlcg_selector(
                    q_model=q_model,
                    training_samples=samples,
                    device=device,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    progress_callback=_on_train_progress,
                )
            else:
                metrics = _train_rlcg_dqn(
                    q_model=q_model,
                    problems=problems,
                    orchestrator=orchestrator,
                    config=config,
                    candidate_source=args.candidate_source,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    device=device,
                    progress_callback=_on_train_progress,
                )
                sample_count_for_logs = int(metrics.get("num_samples", 0.0))
        else:
            samples = _collect_ffcg_samples(
                problems=problems,
                orchestrator=orchestrator,
                config=config,
                candidate_source=args.candidate_source,
                max_iterations=args.max_iterations,
                max_family_size=int(config.get("column_selector", {}).get("max_family_size", 5)),
                tb_tracker=tracker,
            )
            sample_count_for_logs = len(samples)

            def _on_train_progress(train_stats: Dict[str, float]) -> None:
                nonlocal latest_checkpoint_path, latest_checkpoint_epoch, last_logged_epoch, sample_count_for_logs
                local_epoch_index = int(train_stats.get("epoch", 0))
                absolute_epoch_index = resume_base_epoch + local_epoch_index
                last_logged_epoch = absolute_epoch_index
                sample_count_for_logs = int(train_stats.get("num_samples", sample_count_for_logs))
                _append_epoch_metrics_csv_row(
                    metrics_csv_path=metrics_csv_path,
                    absolute_epoch_index=absolute_epoch_index,
                    train_stats=train_stats,
                    num_samples=len(samples),
                    method=args.method,
                    candidate_source=args.candidate_source,
                    training_mode=effective_training_mode,
                )
                tracker.log_metrics(train_stats, step=absolute_epoch_index, prefix="train/ffcg")
                if absolute_epoch_index % args.checkpoint_interval == 0:
                    latest_checkpoint_path = _save_epoch_checkpoint(absolute_epoch_index, train_stats)
                    latest_checkpoint_epoch = absolute_epoch_index
                    logger.info("Saved selector checkpoint: %s", latest_checkpoint_path.name)

            metrics = train_ffcg_selector(
                q_model=q_model,
                training_samples=samples,
                device=device,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                progress_callback=_on_train_progress,
            )
        elapsed_seconds = time.time() - train_start_time
        if latest_checkpoint_path is None or latest_checkpoint_epoch != max(1, last_logged_epoch):
            latest_checkpoint_path = _save_epoch_checkpoint(max(1, last_logged_epoch), metrics)
            logger.info("Saved selector checkpoint: %s", latest_checkpoint_path.name)

        logger.info(
            "Selector training finished: method=%s  samples=%d  elapsed=%.1fs",
            args.method,
            sample_count_for_logs,
            elapsed_seconds,
        )
        final_metric_step = max(1, int(last_logged_epoch))
        tracker.log_metrics(metrics, step=final_metric_step, prefix=f"final/{args.method}")
        tracker.log_scalar("final/elapsed_seconds", float(elapsed_seconds), step=final_metric_step)

        output_path = latest_checkpoint_path
        if args.output.strip():
            output_path = _resolve_output_checkpoint_path(
                output_argument=args.output,
                default_checkpoint_dir=ckpt_dir,
                default_checkpoint_name=latest_checkpoint_path.name,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": q_model.state_dict(),
                    "epoch": int(max(1, last_logged_epoch)),
                    "method": args.method,
                    "training_mode": effective_training_mode,
                    "candidate_source": args.candidate_source,
                    "train_metrics": metrics,
                    "num_samples": sample_count_for_logs,
                    "run_folder": run_folder,
                    "resume_from": resume_checkpoint_path or None,
                },
                str(output_path),
            )
            logger.info("Saved extra output checkpoint: %s", output_path)
        summary_json_path = os.path.join(run_folder, "selector_training_summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as summary_handle:
            json.dump(
                {
                    "method": args.method,
                    "training_mode": effective_training_mode,
                    "candidate_source": args.candidate_source,
                    "max_instances": args.max_instances,
                    "max_iterations": args.max_iterations,
                    "epochs": args.epochs,
                    "checkpoint_interval": args.checkpoint_interval,
                    "learning_rate": args.learning_rate,
                    "num_samples": sample_count_for_logs,
                    "elapsed_seconds": elapsed_seconds,
                    "train_metrics": metrics,
                    "checkpoint_path": str(output_path),
                    "latest_epoch_checkpoint": str(latest_checkpoint_path),
                    "last_epoch": int(max(1, last_logged_epoch)),
                    "run_folder": run_folder,
                },
                summary_handle,
                indent=2,
            )
        print(f"Trained {args.method} selector with {sample_count_for_logs} samples.")
        print(f"Saved checkpoint: {output_path}")
        print(f"Run folder: {run_folder}")
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
