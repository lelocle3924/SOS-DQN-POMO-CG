"""Lightweight DQN-style route selector for pricing columns.

This module keeps memory use low and is designed for CPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class RouteSample:
    vehicle_type: str
    num_customers: int
    total_cost: float
    distance_km: float
    time_hours: float
    reduced_cost: float


class DQNSelector(nn.Module):
    """Small MLP for binary keep/discard decisions."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def build_feature_vector(
    sample: RouteSample,
    vehicle_to_index: Dict[str, int],
    vehicle_count: int,
) -> np.ndarray:
    """Create normalized feature vector with vehicle one-hot."""
    one_hot = np.zeros(vehicle_count, dtype=np.float32)
    one_hot[vehicle_to_index[sample.vehicle_type]] = 1.0
    numeric = np.array(
        [
            float(sample.num_customers),
            float(sample.total_cost),
            float(sample.distance_km),
            float(sample.time_hours),
            float(sample.reduced_cost),
        ],
        dtype=np.float32,
    )
    return np.concatenate([numeric, one_hot], axis=0)


def normalize_features(feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z-scored features and normalization stats."""
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (feature_matrix - mean) / std
    return normalized, mean, std


def train_selector(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 1.0e-3,
    device: torch.device | None = None,
) -> Tuple[DQNSelector, np.ndarray, np.ndarray]:
    """Train selector with BCE objective."""
    if device is None:
        device = torch.device("cpu")

    normalized_features, mean, std = normalize_features(feature_matrix)
    x = torch.tensor(normalized_features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    model = DQNSelector(input_dim=x.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    num_samples = x.shape[0]
    for _ in range(num_epochs):
        permutation = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            idx = permutation[start:end]
            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, mean, std


def score_routes(
    model: DQNSelector,
    samples: Iterable[RouteSample],
    vehicle_to_index: Dict[str, int],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device | None = None,
) -> List[float]:
    """Return keep probability for each route sample."""
    if device is None:
        device = torch.device("cpu")
    model.eval()
    features = np.stack(
        [
            build_feature_vector(sample, vehicle_to_index, len(vehicle_to_index))
            for sample in samples
        ],
        axis=0,
    )
    normalized = (features - mean) / std
    tensor_x = torch.tensor(normalized, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
    return probs
