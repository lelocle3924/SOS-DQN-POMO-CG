# Reproducibility Guide (POMO-CG + RLCG/FFCG)

This repository supports:

- POMO-based pricing in Column Generation (POMO-CG)
- RLCG-style selector (single-column policy)
- FFCG-style selector (variable-size family policy)

The guide below shows exactly:

- which file to run
- which config fields to set
- which terminal commands to use

---

## 0) Environment Setup

### 0.1 Open terminal in project root

```powershell
cd "D:\Uni Material\Code_Odyssey\Code\DQN\260407"
```

### 0.2 Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 1) Base Configuration

Main config file: `configs/default_config.py`

Recommended CPU-safe baseline:

```python
config = {
    "solver": {
        "device": "cpu",
        "seed": 42,
    },

    "training": {
        "batch_size": 1,
        # ... other training fields
    },

    "pomo": {
        "num_pomo_starts": 10,
        # ... other pomo fields
    },

    "column_selector": {
        "method": "none",      # one of: none, rlcg, ffcg
        "checkpoint": "",      # fill later when using a trained selector
        "hidden_dim": 64,
        "max_family_size": 5,
        "stop_q_threshold": 0.0,
    },

    "rlcg_training": {
        "alpha": 1.0,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 1000,
        "replay_capacity": 5000,
        "min_replay_size": 128,
        "batch_size": 32,
        "train_steps_per_collect": 10,
        "target_update_interval": 50,
        "max_episode_steps": 8,
        "gradient_clip_norm": 1.0,
    },
    # ... other config components
}
```

---

## 2) Train POMO-CG

Script to run: `train_pomo.py`

```powershell
python train_pomo.py --config configs/default_config.py --run-name "pomo_mixed_depots"
```

Enable TensorBoard logging:

```powershell
python train_pomo.py --config configs/default_config.py --run-name "pomo_tb" --tensorboard
```

Enable Weights & Biases logging (optional):

```powershell
python train_pomo.py `
  --config configs/default_config.py `
  --run-name "pomo_wandb" `
  --wandb `
  --wandb-project "pomo-cg" `
  --wandb-tags "colab,vrptw,pomo"
```

Useful overrides:

```powershell
python train_pomo.py `
  --config configs/default_config.py `
  --run-name "pomo_quick" `
  --num-epochs 20 `
  --num-customers 40 `
  --results-dir results
```

Outputs are stored in a run folder under `results`, typically:

- `results\train_...\pretrained_pomo\best_model.pt` 
- `results\train_...\pretrained_pomo\final_model.pt`

---

## 3) Test Run POMO-CG

### 3.1 Root-node CG sanity run

Script to run: `run_inference.py`

```powershell
python run_inference.py `
  --config configs/default_config.py `
  --checkpoint "D:\...\results\train_...\pretrained_pomo\final_model.pt" `
  --max-iterations 30 `
  --verbose
```

Add only the most negative reduced-cost column each iteration:

```powershell
python run_inference.py `
  --config configs/default_config.py `
  --checkpoint "D:\...\results\train_...\pretrained_pomo\final_model.pt" `
  --add-most-negative
```

### 3.2 Full Branch-and-Price run (POMO only)

Set in config:

```python
# configs/default_config.py
    "column_selector": {
        "method": "none",
        # ...
    }
```

Script to run: `main.py`

```powershell
python main.py --config configs/default_config.py --run-name "bp_pomo_only"
```

---

## 4) Train RLCG Selector (Single-Column Policy)

Script to run: `train_column_selectors.py`

Candidate source can be:

- `pomo`
- `ga`
- `dp`

Training mode can be:

- `dqn` (default, paper-aligned RL training with replay buffer + target network)
- `imitation` (legacy teacher-label baseline)

### 4.1 RLCG with POMO candidates (DQN mode)

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method rlcg `
  --training-mode dqn `
  --candidate-source pomo `
  --max-instances 4 `
  --epochs 10 `
  --checkpoint-interval 5 `
  --tensorboard
```

### 4.2 RLCG imitation baseline (legacy)

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method rlcg `
  --training-mode imitation `
  --candidate-source pomo `
  --max-instances 8 `
  --max-iterations 4 `
  --epochs 20
```

---

### 4.3 Extensive config overrides from CLI (Colab-friendly)

Use `--set key.path=value` repeatedly to avoid editing/uploading config for each run:

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method rlcg `
  --training-mode dqn `
  --candidate-source pomo `
  --set solver.device=cpu `
  --set rlcg_training.alpha=1.0 `
  --set rlcg_training.gamma=0.98 `
  --set rlcg_training.epsilon_start=1.0 `
  --set rlcg_training.epsilon_end=0.1 `
  --set rlcg_training.epsilon_decay_steps=500 `
  --set rlcg_training.replay_capacity=2000 `
  --set rlcg_training.min_replay_size=64 `
  --set rlcg_training.batch_size=16 `
  --set rlcg_training.train_steps_per_collect=5 `
  --set rlcg_training.max_episode_steps=6
```

---

### 4.4 CPU / 8GB RAM quick smoke run

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method rlcg `
  --training-mode dqn `
  --candidate-source dp `
  --max-instances 2 `
  --epochs 2 `
  --set solver.device=cpu `
  --set rlcg_training.replay_capacity=512 `
  --set rlcg_training.min_replay_size=32 `
  --set rlcg_training.batch_size=8 `
  --set rlcg_training.train_steps_per_collect=2 `
  --set rlcg_training.max_episode_steps=4
```

---

## 5) Train FFCG Selector (Variable-Size Family Policy)

Use the same script with `--method ffcg`.

### 5.1 FFCG with POMO candidates

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method ffcg `
  --candidate-source pomo `
  --max-instances 8 `
  --max-iterations 4 `
  --epochs 20 `
  --checkpoint-interval 5 \
  --tensorboard
```

### 5.2 FFCG with GA candidates

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method ffcg `
  --candidate-source ga `
  --max-instances 8 `
  --max-iterations 4 `
  --epochs 20 \
  --checkpoint-interval 5 \
```

### 5.3 FFCG with DP candidates

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method ffcg `
  --candidate-source dp `
  --max-instances 8 `
  --max-iterations 4 `
  --epochs 20 \
  --checkpoint-interval 5 \
```

### 5.4 Continue training a selector (resume)

You can continue training from an existing selector checkpoint using `--resume-from`.
Selector checkpoints are saved each epoch with POMO-style numbering:

- `pretrained_col_selector/dqn_model_rlcg_epoch_<N>.pt`
- `pretrained_col_selector/dqn_model_ffcg_epoch_<N>.pt`

When resuming, training continues epoch numbering from the loaded checkpoint epoch.

Resume RLCG example:

```powershell
python train_column_selectors.py \
  --config configs/default_config.py \
  --method rlcg \
  --candidate-source ga \
  --max-instances 8 \
  --max-iterations 4 \
  --epochs 10 \
  --resume-from "D:\path\to\previous\dqn_model_rlcg_epoch_20.pt" \
  --checkpoint-interval 5 \
  --tensorboard
```

Resume FFCG example:

```powershell
python train_column_selectors.py `
  --config configs/default_config.py `
  --method ffcg `
  --candidate-source pomo `
  --max-instances 8 `
  --max-iterations 4 `
  --epochs 10 `
  --resume-from "D:\path\to\previous\dqn_model_ffcg_epoch_20.pt" `
  --checkpoint-interval 5 \
  --tensorboard
```

Resume in the same existing run folder:

```powershell
python train_column_selectors.py \
  --config configs/default_config.py \
  --method rlcg \
  --candidate-source ga \
  --epochs 10 \
  --resume-dir "D:\...\results\train_YYYYMMDD_HHMMSS_col_selector_rlcg" \
  --checkpoint-interval 5
```

Notes:

- `--method` must match the checkpoint type you resume from.
- `--resume-dir <existing_run_folder>` resumes in the same folder (POMO-style), continues epoch numbering, and appends `training_metrics.csv`.
- Without `--resume-dir`, training creates a new run folder (`train_YYYYMMDD_HHMMSS_col_selector_<method>`).
- Epoch checkpoints are always saved under that run folder in `pretrained_col_selector`.
- `training_metrics.csv` logs one row per epoch, so you can directly plot DQN learning curves.
- `--output` is optional and saves an additional checkpoint copy (usually the latest epoch).

---

## 6) Use Pretrained POMO-CG + Pretrained RLCG/FFCG Together

Edit `configs/default_config.py`:

```python
config = {
    # ...
    "training": {
        # ...
        "pretrained_model": "D:/.../pretrained_pomo/final_model.pt",
    },

    "column_selector": {
        "method": "rlcg",  # or "ffcg"
        "checkpoint": "D:/.../results/train_.../pretrained_col_selector/dqn_model_rlcg_epoch_20.pt",  # or ffcg checkpoint
        "hidden_dim": 64,
        "max_family_size": 5,
        "stop_q_threshold": 0.0,
    },
    # ...
}
```

Then run:

```powershell
python main.py --config configs/default_config.py --run-name "bp_pomo_plus_selector"
```

---

## 7) Suggested Experiment Matrix

For fair comparison, run:

- POMO only (`method=none`)
- POMO + RLCG (`method=rlcg`)
- POMO + FFCG (`method=ffcg`)

For each selector, train with each candidate source:

- `pomo`
- `ga`
- `dp`

Track:

- CG iterations
- CPU time
- LP objective
- Integer objective (if branch-and-price run completes)

---

## 8) Quick Command Summary

Train POMO:

```powershell
python train_pomo.py --config configs/default_config.py --run-name "pomo_mixed_depots"
```

Test POMO root CG:

```powershell
python run_inference.py --config configs/default_config.py --checkpoint "D:\...\final_model.pt" --verbose
```

Train RLCG:

```powershell
python train_column_selectors.py --config configs/default_config.py --method rlcg --training-mode dqn --candidate-source pomo
```

Train FFCG:

```powershell
python train_column_selectors.py --config configs/default_config.py --method ffcg --candidate-source pomo
```

Run full B&P with pretrained models:

```powershell
python main.py --config configs/default_config.py --run-name "bp_final"
```

### 4.3 Extensive config overrides from CLI (Colab-friendly)

Use `--set key.path=value` repeatedly to avoid editing/uploading config for each run: