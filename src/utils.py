"""Utility functions: seed fixing, time parsing, normalization."""

import random
import numpy as np
import torch


def fix_all_seeds(seed: int) -> None:
    """Fix random seeds for numpy, torch, and python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_time_string(time_str: str) -> float:
    """Convert 'HH:MM' or 'H:MM' string to fractional hours. E.g., '8:30' -> 8.5."""
    parts = time_str.strip().split(":")
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    return hours + minutes / 60.0


def parse_allowed_trucks(truck_str: str) -> set:
    """Parse '{AUV, MC, 4w, 6w}' into a set of truck name strings."""
    if not isinstance(truck_str, str):
        return set()
    cleaned = truck_str.strip().strip("{}")
    return {token.strip() for token in cleaned.split(",")}


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Min-max normalize an (N, 2) coordinate array to [0, 1]."""
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    return (coords - min_vals) / range_vals
