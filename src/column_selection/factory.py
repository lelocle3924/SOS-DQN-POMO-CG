"""Factory to instantiate column selectors from config."""

from __future__ import annotations

import os
from typing import Optional

import torch

from src.column_selection.base import AbstractColumnSelector
from src.column_selection.ffcg_selector import FFCGSelector
from src.column_selection.rlcg_selector import RLCGSelector
from src.graph_dqn import BipartiteGraphQNetwork


def build_column_selector(config: dict, device: torch.device) -> Optional[AbstractColumnSelector]:
    selector_cfg = config.get("column_selector", {})
    method = str(selector_cfg.get("method", "none")).strip().lower()
    if method in {"none", ""}:
        return None

    model = BipartiteGraphQNetwork(
        column_feature_dim=9,
        constraint_feature_dim=2,
        hidden_dim=int(selector_cfg.get("hidden_dim", 64)),
    )

    checkpoint_path = selector_cfg.get("checkpoint", "")
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rc_tolerance = float(config["column_generation"].get("reduced_cost_tolerance", -1e-6))
    if method == "rlcg":
        return RLCGSelector(model, device=device, rc_tolerance=rc_tolerance)
    if method == "ffcg":
        return FFCGSelector(
            model,
            device=device,
            rc_tolerance=rc_tolerance,
            max_family_size=int(selector_cfg.get("max_family_size", 5)),
            stop_q_threshold=float(selector_cfg.get("stop_q_threshold", 0.0)),
        )
    raise ValueError(f"Unsupported column_selector.method: {method}")
