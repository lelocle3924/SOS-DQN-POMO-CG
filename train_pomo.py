"""POMO pre-training entry-point.

Trains the Attention Model on random VRPTW instances using REINFORCE
with the POMO multi-start baseline.
"""

import argparse
import time
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import fix_all_seeds, load_config
from src.data_loader import (
    build_training_manifest,
    load_problem,
    load_problem_for_training_instance,
)
from src.pomo_model import POMOModel
from src.pomo_trainer import (
    InstanceGenerator,
    build_node_features,
    run_pomo_rollout,
    train_epoch,
)
from src.pomo_env import VRPTWEnvironment
from src.run_manager import create_run_folder, setup_logging, shadow_copy_config

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Optional TensorBoard + Weights & Biases logging."""

    def __init__(
        self,
        run_folder: str,
        config: Dict[str, Any],
        use_tensorboard: bool,
        use_wandb: bool,
        wandb_project: str,
        wandb_entity: Optional[str],
        wandb_run_name: str,
        wandb_tags: List[str],
    ) -> None:
        self._tb_writer = None
        self._wandb = None
        self._wandb_enabled = False

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = os.path.join(run_folder, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=tb_dir)
                logger.info("TensorBoard logging enabled at: %s", tb_dir)
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("TensorBoard unavailable (%s). Continuing without it.", exc)

        if use_wandb:
            try:
                import wandb

                init_kwargs: Dict[str, Any] = {
                    "project": wandb_project,
                    "name": wandb_run_name,
                    "dir": run_folder,
                    "config": config,
                    "reinit": True,
                }
                if wandb_entity:
                    init_kwargs["entity"] = wandb_entity
                if wandb_tags:
                    init_kwargs["tags"] = wandb_tags
                wandb.init(**init_kwargs)
                self._wandb = wandb
                self._wandb_enabled = True
                logger.info("Weights & Biases logging enabled for project '%s'.", wandb_project)
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning(
                    "wandb unavailable or init failed (%s). Continuing without wandb.", exc
                )

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self._tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, step)
        if self._wandb_enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
        if self._wandb_enabled and self._wandb is not None:
            self._wandb.finish()


def _load_mixed_problem_pool(config: dict):
    training_data = config.get("training_data", {})
    temp_day_dirs = [
        os.path.join(PROJECT_ROOT, p) for p in training_data.get("temp_day_dirs", [])
    ]
    distance_matrix_dirs = [
        os.path.join(PROJECT_ROOT, p) for p in training_data.get("distance_matrix_dirs", [])
    ]
    if not temp_day_dirs or not distance_matrix_dirs:
        return [], []

    manifest = build_training_manifest(
        temp_day_dirs=temp_day_dirs,
        distance_matrix_dirs=distance_matrix_dirs,
        distance_matrix_pattern=training_data.get(
            "distance_matrix_pattern", "distance_matrix_meters{depot_id}.csv"
        ),
    )

    trucks_path = os.path.join(PROJECT_ROOT, config["problem"]["trucks_file"])
    loaded_problems = []
    for spec in manifest:
        loaded_problems.append(
            load_problem_for_training_instance(spec, trucks_path=trucks_path)
        )
    return loaded_problems, manifest


def _validate(
    model: POMOModel,
    instance_gen: InstanceGenerator,
    config: dict,
    device: torch.device,
) -> float:
    """Validation pass with greedy decoding. Returns mean reward."""
    model.eval()
    batch_size = config["training"]["batch_size"]
    num_cust = config["training"]["max_customers_per_instance"]
    num_pomo = config["pomo"]["num_pomo_starts"]
    cap = 1000.0
    tw_end = config["problem"].get("planning_horizon_hours", 24.0)
    env = VRPTWEnvironment(device)

    num_batches = max(1, config["training"]["num_val_instances"] // batch_size)
    total = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            inst = instance_gen.generate_batch(batch_size, num_cust)
            batch_cap = inst.get("vehicle_capacity", cap)
            res = run_pomo_rollout(
                model, env, inst, num_pomo, batch_cap, tw_end,
                decode_method="greedy",
                fixed_cost=inst.get("fixed_cost", 10.0),
                cost_per_km=inst.get("cost_per_km", 1.0),
                cost_per_hour=inst.get("cost_per_hour", 1.0),
            )
            total += res["rewards"].max(dim=1).values.mean().item()

    model.train()
    return total / num_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="POMO Pre-training")
    parser.add_argument("--config", default="configs/default_config.py")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--resume-dir", type=str, default=None, help="Path to a previous run folder to resume from")
    parser.add_argument("--num-customers", type=int, default=None, help="Override max_customers_per_instance")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch_size")
    parser.add_argument("--num-pomo-starts", type=int, default=None, help="Override pomo num_pomo_starts")
    parser.add_argument("--run-name", type=str, default="pomo_training", help="Name for the run folder")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results_dir from config")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="pomo-cg", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated wandb tags, e.g. colab,stage1,pomo",
    )
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    if args.num_customers is not None:
        if args.num_customers <= 0:
            raise ValueError("--num-customers must be a positive integer.")
        config["training"]["max_customers_per_instance"] = args.num_customers
    if args.num_epochs is not None:
        if args.num_epochs <= 0:
            raise ValueError("--num-epochs must be a positive integer.")
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be a positive integer.")
        config["training"]["batch_size"] = args.batch_size
    if args.num_pomo_starts is not None:
        if args.num_pomo_starts <= 0:
            raise ValueError("--num-pomo-starts must be a positive integer.")
        config["pomo"]["num_pomo_starts"] = args.num_pomo_starts
    if args.results_dir is not None:
        config["logging"]["results_dir"] = args.results_dir

    fix_all_seeds(config["solver"]["seed"])

    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    if args.resume_dir and os.path.isdir(args.resume_dir):
        run_folder = args.resume_dir
        # Setup logging in the existing folder
        setup_logging(run_folder, config["logging"]["log_level"])
        logger.info("Resuming in existing run folder: %s", run_folder)
    else:
        # Create timestamped folder starting with 'train_'
        from datetime import datetime
        import pytz
        vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        timestamp = datetime.now(vn_tz).strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(results_dir, f"train_{timestamp}_{args.run_name}")
        os.makedirs(run_folder, exist_ok=True)
        shadow_copy_config(args.config, run_folder)
        setup_logging(run_folder, config["logging"]["log_level"])

    device_name = config["solver"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    logger.info("Training device: %s", device)

    wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    tracker = ExperimentTracker(
        run_folder=run_folder,
        config=config,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name or os.path.basename(run_folder),
        wandb_tags=wandb_tags,
    )

    model = POMOModel(
        node_feature_dim=config["pomo"]["node_feature_dim"],
        embedding_dim=config["pomo"]["embedding_dim"],
        num_heads=config["pomo"]["num_heads"],
        num_encoder_layers=config["pomo"]["num_encoder_layers"],
        ff_dim=config["pomo"]["feedforward_dim"],
    ).to(device)

    if args.resume_from and os.path.isfile(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Loaded pre-trained weights from: %s", args.resume_from)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    start_epoch = 1
    best_val = -float("inf")
    
    # Setup CSV logging
    import csv
    
    if args.resume_dir and os.path.isdir(args.resume_dir):
        run_folder = args.resume_dir
        ckpt_dir = os.path.join(run_folder, "pretrained_pomo")
        csv_path = os.path.join(run_folder, "training_metrics.csv")
        
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch_") and f.endswith(".pt")]
        ckpt_path = None
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda x: int(x.replace("model_epoch_", "").replace(".pt", "")))
            ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
        elif os.path.exists(os.path.join(ckpt_dir, "best_model.pt")):
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            torch.set_rng_state(checkpoint["rng_state"].cpu().byte())
            if "cuda_rng_state" in checkpoint and checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu().byte())
            start_epoch = checkpoint["epoch"] + 1
            best_val = checkpoint.get("best_val", -float("inf"))
            logger.info("Resumed training from %s (starting at epoch %d)", ckpt_path, start_epoch)
            
            # Clean up CSV to prevent duplicate epoch rows
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    lines = list(csv.reader(f))
                
                # Keep header and rows up to the checkpoint epoch
                cleaned_lines = [lines[0]]
                for row in lines[1:]:
                    if row and row[0].isdigit() and int(row[0]) <= checkpoint["epoch"]:
                        cleaned_lines.append(row)
                
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(cleaned_lines)
                    
        else:
            logger.warning("No checkpoints found in %s to resume from.", ckpt_dir)
            
    else:
        ckpt_dir = os.path.join(run_folder, "pretrained_pomo")
        os.makedirs(ckpt_dir, exist_ok=True)
        csv_path = os.path.join(run_folder, "training_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "loss",
                "avg_reward",
                "baseline_variance",
                "penalty_ratio",
                "max_memory_load_percent",
                "val_reward",
            ])

    loaded_problems = []
    manifest = []
    try:
        loaded_problems, manifest = _load_mixed_problem_pool(config)
    except Exception as exc:
        logger.warning("Mixed problem pool unavailable (%s).", exc)

    if loaded_problems:
        logger.info(
            "Loaded mixed training pool: %d instances from %d depots.",
            len(loaded_problems),
            len({spec.depot_id for spec in manifest}),
        )
        instance_gen = InstanceGenerator(
            config, device, problems=loaded_problems, instance_specs=manifest
        )
    else:
        problem = None
        try:
            problem = load_problem(
                orders_path=os.path.join(PROJECT_ROOT, config["problem"]["orders_file"]),
                trucks_path=os.path.join(PROJECT_ROOT, config["problem"]["trucks_file"]),
                distance_matrix_path=os.path.join(
                    PROJECT_ROOT, config["problem"]["distance_matrix_file"]
                ),
                depot_id=config["problem"]["depot_id"],
            )
            logger.info(
                "Loaded single real problem for sub-graph training (%d customers).",
                problem.num_customers,
            )
        except Exception as exc:
            logger.warning("Could not load real problem data (%s). Using synthetic.", exc)
        instance_gen = InstanceGenerator(config, device, problem=problem)
    num_epochs = config["training"]["num_epochs"]
    checkpoint_interval = config["training"].get("checkpoint_interval", 50)

    training_data_cfg = config.get("training_data", {})
    stage_plan = training_data_cfg.get(
        "curriculum_stages",
        [
            {"name": "tiny_warmup", "epochs": min(2, num_epochs), "max_customers": 8},
            {"name": "main", "epochs": max(0, num_epochs - min(2, num_epochs)),
             "max_customers": config["training"]["max_customers_per_instance"]},
        ],
    )
    if args.num_customers is not None:
        capped_stage_plan = []
        for stage in stage_plan:
            stage_copy = dict(stage)
            stage_copy["max_customers"] = min(
                int(stage_copy.get("max_customers", args.num_customers)),
                args.num_customers,
            )
            capped_stage_plan.append(stage_copy)
        stage_plan = capped_stage_plan

    global_epoch = start_epoch - 1
    train_start_time = time.time()
    total_epochs_to_run = max(1, num_epochs - (start_epoch - 1))
    try:
        for stage in stage_plan:
            stage_name = stage.get("name", "main")
            stage_epochs = int(stage.get("epochs", 0))
            if stage_epochs <= 0:
                continue
            stage_max_customers = int(
                stage.get("max_customers", config["training"]["max_customers_per_instance"])
            )
            config["training"]["max_customers_per_instance"] = stage_max_customers
            instance_gen.set_stage(stage_name)

            logger.info(
                "\n=== Curriculum stage: %s  epochs=%d  max_customers=%d ===",
                stage_name, stage_epochs, stage_max_customers
            )

            for _ in range(stage_epochs):
                global_epoch += 1
                logger.info("\n--- Epoch %d / %d ---", global_epoch, num_epochs)
                metrics = train_epoch(model, optimizer, instance_gen, config, global_epoch)
                logger.info(
                    "Epoch %d  stage=%s  loss=%.6f  reward=%.2f  var=%.2f  penalty=%.2f%%  mem_max=%.1f%%",
                    global_epoch, stage_name, metrics["loss"], metrics["reward"],
                    metrics["variance"], metrics["penalty_ratio"] * 100,
                    metrics.get("max_memory_load_percent", float("nan")),
                )
                completed_epochs = max(1, global_epoch - (start_epoch - 1))
                epoch_progress = completed_epochs / total_epochs_to_run
                elapsed_seconds = time.time() - train_start_time
                eta_seconds = (elapsed_seconds / epoch_progress) - elapsed_seconds if epoch_progress > 0 else float("nan")
                logger.info(
                    "Training progress: %d/%d (%.1f%%)  elapsed=%.1fs  eta=%.1fs",
                    completed_epochs,
                    total_epochs_to_run,
                    epoch_progress * 100.0,
                    elapsed_seconds,
                    eta_seconds,
                )

                val_reward = None
                if global_epoch % 5 == 0:
                    stage_names_for_validation = list({
                        s.get("name", "main") for s in stage_plan
                    })
                    stage_rewards: List[float] = []
                    for val_stage_name in stage_names_for_validation:
                        instance_gen.set_stage(val_stage_name)
                        stage_rewards.append(_validate(model, instance_gen, config, device))
                    instance_gen.set_stage(stage_name)
                    val_reward = float(sum(stage_rewards) / len(stage_rewards))
                    logger.info("Validation reward (all stages): %.4f", val_reward)
                    if val_reward > best_val:
                        best_val = val_reward
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": global_epoch,
                            "curriculum_stage": stage_name,
                            "rng_state": torch.get_rng_state(),
                            "best_val": best_val,
                            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        }, os.path.join(ckpt_dir, "best_model.pt"))
                        logger.info("Saved best model (val_reward=%.4f)", val_reward)

                tracker.log(
                    {
                        "train/loss": float(metrics["loss"]),
                        "train/reward": float(metrics["reward"]),
                        "train/variance": float(metrics["variance"]),
                        "train/penalty_ratio": float(metrics["penalty_ratio"]),
                        "system/max_memory_load_percent": float(
                            metrics.get("max_memory_load_percent", float("nan"))
                        ),
                        "train/stage_max_customers": float(stage_max_customers),
                        "train/batch_size": float(config["training"]["batch_size"]),
                        "train/num_pomo_starts": float(config["pomo"]["num_pomo_starts"]),
                    },
                    step=global_epoch,
                )
                if val_reward is not None:
                    tracker.log({"val/reward": float(val_reward), "val/best_reward": float(best_val)}, step=global_epoch)

                if global_epoch % checkpoint_interval == 0:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": global_epoch,
                        "curriculum_stage": stage_name,
                        "rng_state": torch.get_rng_state(),
                        "best_val": best_val,
                    }, os.path.join(ckpt_dir, f"model_epoch_{global_epoch}.pt"))

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        global_epoch,
                        metrics["loss"],
                        metrics["reward"],
                        metrics["variance"],
                        metrics["penalty_ratio"],
                        metrics.get("max_memory_load_percent", float("nan")),
                        val_reward if val_reward is not None else "",
                    ])

                if global_epoch >= num_epochs:
                    break
            if global_epoch >= num_epochs:
                break

        final_path = os.path.join(ckpt_dir, "final_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": max(global_epoch, 0),
            "curriculum_stage": args.run_name,
            "rng_state": torch.get_rng_state(),
            "best_val": best_val,
        }, final_path)
        logger.info("Training complete. Final model → %s", final_path)
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
