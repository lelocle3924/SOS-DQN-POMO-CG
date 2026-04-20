"""Run management: timestamped folders, config shadow copy, logging setup."""

import os
import shutil
import json
import csv
import logging
from datetime import datetime
from typing import Dict


def create_run_folder(results_dir: str, run_name: str = "") -> str:
    """Create a timestamped folder inside results_dir. Returns its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_path = os.path.join(results_dir, folder_name)
    os.makedirs(run_path, exist_ok=True)
    return run_path


def shadow_copy_config(config_path: str, run_folder: str) -> str:
    """Copy the active config file into the run folder for reproducibility."""
    dest = os.path.join(run_folder, os.path.basename(config_path))
    shutil.copy2(config_path, dest)
    return dest


def setup_logging(run_folder: str, log_level: str = "INFO") -> str:
    """Configure logging to both console and a file inside run_folder."""
    log_path = os.path.join(run_folder, "run.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return log_path


def save_raw_output(data: Dict, run_folder: str,
                    filename: str = "raw_output.json") -> str:
    """Serialise raw solver output to a JSON file inside the run folder."""
    path = os.path.join(run_folder, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path


def append_to_master_csv(results_dir: str, run_id: str,
                         metrics: Dict) -> None:
    """Append a single-row summary to the master results.csv."""
    csv_path = os.path.join(results_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)

    row = {"run_id": run_id, **metrics}
    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
