#!/usr/bin/env python3
"""
evaluate_cql_agent.py

Evaluate a trained discrete Conservative Q-Learning (CQL) baseline for the
same temporal sepsis treatment optimization setup used for PPO and A2C.

This version is aligned with:
    1. sepsis_temporal_env.py
    2. prepare_offline_dataset_env_aligned.py
    3. train_cql_agent.py

It expects the offline dataset produced by prepare_offline_dataset_env_aligned.py
and the CQL checkpoint produced by the updated train_cql_agent.py.

It does NOT perform online clinical simulation. It evaluates the learned CQL
Q-network on the fixed retrospective transition dataset and produces offline
policy diagnostics, including:

    - CQL greedy action distribution
    - Dataset / clinician action distribution
    - Fluid and vasopressor intensity summaries
    - Q-value summaries
    - Bellman / TD-error diagnostics
    - Direct-method-style estimated value from learned Q-values
    - Observed dataset / clinician trajectory returns
    - Severity subgroup analysis using sofa_proxy from env metadata, if available

Expected NPZ arrays:
    states       : shape [N, state_dim]
    actions      : shape [N]
    rewards      : shape [N]
    next_states  : shape [N, state_dim]
    dones        : shape [N]

Optional NPZ arrays:
    stay_ids     : shape [N]
    timesteps    : shape [N]

Main outputs:
    cql_evaluation_metrics.json
    cql_transition_predictions.csv
    cql_eval_action_distribution.csv
    cql_action_heatmap.csv
    cql_severity_subgroup_policy.csv, if possible
    cql_q_value_summary.csv
    cql_vs_dataset_action_distribution.png
    cql_action_heatmap.png
    cql_severity_vasopressor.png, if possible
    cql_q_value_histogram.png
    cql_td_error_histogram.png
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Model definition
# Must match train_cql_agent.py
# ---------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    MLP Q-network for discrete action CQL.
    Outputs Q-values for all treatment actions.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 25,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = state_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------

def load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path, allow_pickle=True)

    required = ["states", "actions", "rewards", "next_states", "dones"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(
            f"Missing required arrays in NPZ file: {missing}. Available arrays: {data.files}"
        )

    dataset = {key: data[key] for key in data.files}
    dataset["states"] = dataset["states"].astype(np.float32)
    dataset["actions"] = dataset["actions"].astype(np.int64)
    dataset["rewards"] = dataset["rewards"].astype(np.float32)
    dataset["next_states"] = dataset["next_states"].astype(np.float32)
    dataset["dones"] = dataset["dones"].astype(np.float32)

    return dataset


def sanitize_dataset(dataset: Dict[str, np.ndarray], num_actions: int) -> Dict[str, np.ndarray]:
    states = dataset["states"]
    next_states = dataset["next_states"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    dones = dataset["dones"]

    finite_mask = (
        np.isfinite(states).all(axis=1)
        & np.isfinite(next_states).all(axis=1)
        & np.isfinite(actions)
        & np.isfinite(rewards)
        & np.isfinite(dones)
    )
    action_mask = (actions >= 0) & (actions < num_actions)
    keep_mask = finite_mask & action_mask

    removed = int(len(actions) - keep_mask.sum())
    if removed > 0:
        print(f"[WARN] Removed {removed} invalid transitions before evaluation.")

    cleaned: Dict[str, np.ndarray] = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and len(value) == len(actions):
            cleaned[key] = value[keep_mask]
        else:
            cleaned[key] = value

    if len(cleaned["actions"]) == 0:
        raise ValueError("No valid transitions remain after cleaning.")

    return cleaned


def load_metadata(metadata_path: Optional[Path]) -> Dict:
    if metadata_path is None:
        return {}
    if not metadata_path.exists():
        print(f"[WARN] Metadata file not found: {metadata_path}. Severity analysis may be skipped.")
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_torch_load(model_path: Path, device: torch.device) -> Dict:
    """
    Load a PyTorch checkpoint across versions.

    PyTorch versions differ around the weights_only argument. This helper keeps
    the script compatible with older and newer installations.
    """
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def parse_hidden_sizes_from_checkpoint(checkpoint: Dict, fallback: str = "256,256") -> Tuple[int, ...]:
    hidden = checkpoint.get("hidden_sizes", None)

    if hidden is None:
        args = checkpoint.get("args", {})
        hidden = args.get("hidden_sizes", fallback) if isinstance(args, dict) else fallback

    if isinstance(hidden, (list, tuple)):
        return tuple(int(x) for x in hidden)

    if isinstance(hidden, str):
        return tuple(int(x.strip()) for x in hidden.split(",") if x.strip())

    return tuple(int(x.strip()) for x in fallback.split(",") if x.strip())


def load_model(
    model_path: Path,
    state_dim: int,
    num_actions: int,
    device: torch.device,
    fallback_hidden_sizes: str = "256,256",
) -> Tuple[QNetwork, Optional[QNetwork], Dict]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = safe_torch_load(model_path, device=device)

    ckpt_state_dim = int(checkpoint.get("state_dim", state_dim))
    ckpt_num_actions = int(checkpoint.get("num_actions", num_actions))
    dropout = float(checkpoint.get("dropout", 0.0))
    hidden_sizes = parse_hidden_sizes_from_checkpoint(checkpoint, fallback=fallback_hidden_sizes)

    if ckpt_state_dim != state_dim:
        raise ValueError(
            f"Checkpoint state_dim={ckpt_state_dim}, but dataset state_dim={state_dim}. "
            "Make sure you are evaluating the matching env-aligned dataset and model."
        )

    if ckpt_num_actions != num_actions:
        print(
            f"[WARN] Checkpoint num_actions={ckpt_num_actions}, but CLI num_actions={num_actions}. "
            f"Using checkpoint value: {ckpt_num_actions}."
        )
        num_actions = ckpt_num_actions

    q_net = QNetwork(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.eval()

    target_net = None
    if "target_model_state_dict" in checkpoint:
        target_net = QNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(device)
        target_net.load_state_dict(checkpoint["target_model_state_dict"])
        target_net.eval()

    return q_net, target_net, checkpoint


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

@torch.no_grad()
def batched_q_values(
    q_net: QNetwork,
    states: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    q_net.eval()
    outputs = []

    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        batch_states = torch.as_tensor(states[start:end], dtype=torch.float32).to(device)
        q_values = q_net(batch_states).detach().cpu().numpy()
        outputs.append(q_values)

    return np.concatenate(outputs, axis=0)


def action_to_fluid(action: np.ndarray) -> np.ndarray:
    return action // 5


def action_to_vaso(action: np.ndarray) -> np.ndarray:
    return action % 5


def safe_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs / max(probs.sum(), 1e-12)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def make_action_distribution_df(
    dataset_actions: np.ndarray,
    greedy_actions: np.ndarray,
    num_actions: int,
) -> pd.DataFrame:
    dataset_counts = np.bincount(dataset_actions, minlength=num_actions)
    greedy_counts = np.bincount(greedy_actions, minlength=num_actions)

    dataset_total = max(int(dataset_counts.sum()), 1)
    greedy_total = max(int(greedy_counts.sum()), 1)

    rows = []
    for action in range(num_actions):
        rows.append({
            "action": int(action),
            "fluid_bin": int(action // 5),
            "vaso_bin": int(action % 5),
            "dataset_count": int(dataset_counts[action]),
            "dataset_percent": float(dataset_counts[action] / dataset_total),
            "cql_greedy_count": int(greedy_counts[action]),
            "cql_greedy_percent": float(greedy_counts[action] / greedy_total),
            "count_difference_cql_minus_dataset": int(greedy_counts[action] - dataset_counts[action]),
            "percent_difference_cql_minus_dataset": float(
                greedy_counts[action] / greedy_total - dataset_counts[action] / dataset_total
            ),
        })

    return pd.DataFrame(rows)


def make_action_heatmap_df(greedy_actions: np.ndarray, num_actions: int) -> pd.DataFrame:
    counts = np.bincount(greedy_actions, minlength=num_actions)
    total = max(int(counts.sum()), 1)

    rows = []
    for action in range(num_actions):
        rows.append({
            "action": int(action),
            "fluid_bin": int(action // 5),
            "vaso_bin": int(action % 5),
            "count": int(counts[action]),
            "percent": float(counts[action] / total),
        })

    return pd.DataFrame(rows)


def estimate_observed_dataset_episode_returns(dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Summarize observed dataset / clinician trajectory returns.

    This is not the learned CQL policy return. It is the return from the fixed
    offline rewards in the retrospective dataset.
    """
    if "stay_ids" not in dataset:
        return {}

    rewards = dataset["rewards"].astype(float)
    stay_ids = dataset["stay_ids"]
    df = pd.DataFrame({"stay_id": stay_ids, "reward": rewards})
    returns = df.groupby("stay_id")["reward"].sum().values

    if len(returns) == 0:
        return {}

    return {
        "observed_dataset_episode_return_mean": float(np.mean(returns)),
        "observed_dataset_episode_return_std": float(np.std(returns)),
        "observed_dataset_episode_return_min": float(np.min(returns)),
        "observed_dataset_episode_return_q25": float(np.percentile(returns, 25)),
        "observed_dataset_episode_return_median": float(np.median(returns)),
        "observed_dataset_episode_return_q75": float(np.percentile(returns, 75)),
        "observed_dataset_episode_return_max": float(np.max(returns)),
        "observed_dataset_num_episodes": int(len(returns)),
    }


def estimate_direct_method_value(
    dataset: Dict[str, np.ndarray],
    max_q_values: np.ndarray,
) -> Dict[str, float]:
    """
    Direct-method-style policy value from learned Q-values.

    The most interpretable estimate is the mean max_a Q(s_0, a) over the first
    transition of each ICU stay. If timesteps are unavailable, all states are used.
    """
    mask = np.ones(len(max_q_values), dtype=bool)

    if "timesteps" in dataset:
        try:
            mask = dataset["timesteps"].astype(int) == 0
        except Exception:
            mask = np.ones(len(max_q_values), dtype=bool)

    if mask.sum() == 0:
        mask = np.ones(len(max_q_values), dtype=bool)

    initial_values = max_q_values[mask]

    return {
        "dm_estimated_cql_initial_state_value_mean": float(np.mean(initial_values)),
        "dm_estimated_cql_initial_state_value_std": float(np.std(initial_values)),
        "dm_estimated_cql_initial_state_value_min": float(np.min(initial_values)),
        "dm_estimated_cql_initial_state_value_median": float(np.median(initial_values)),
        "dm_estimated_cql_initial_state_value_max": float(np.max(initial_values)),
        "dm_num_initial_states_used": int(len(initial_values)),
    }


def compute_td_diagnostics(
    q_dataset_actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    next_max_q: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    target = rewards + gamma * (1.0 - dones) * next_max_q
    td_error = q_dataset_actions - target
    abs_td_error = np.abs(td_error)

    metrics = {
        "td_error_mean": float(np.mean(td_error)),
        "td_error_std": float(np.std(td_error)),
        "td_error_abs_mean": float(np.mean(abs_td_error)),
        "td_error_abs_median": float(np.median(abs_td_error)),
        "td_error_abs_q95": float(np.percentile(abs_td_error, 95)),
        "bellman_mse": float(np.mean(td_error ** 2)),
    }

    return td_error, metrics


def make_q_value_summary_df(q_values: np.ndarray) -> pd.DataFrame:
    rows = []
    num_actions = q_values.shape[1]

    for action in range(num_actions):
        values = q_values[:, action]
        rows.append({
            "action": int(action),
            "fluid_bin": int(action // 5),
            "vaso_bin": int(action % 5),
            "q_mean": float(np.mean(values)),
            "q_std": float(np.std(values)),
            "q_min": float(np.min(values)),
            "q_q25": float(np.percentile(values, 25)),
            "q_median": float(np.median(values)),
            "q_q75": float(np.percentile(values, 75)),
            "q_max": float(np.max(values)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Severity subgroup analysis
# ---------------------------------------------------------------------

def get_feature_columns_from_metadata(metadata: Dict) -> Optional[List[str]]:
    """
    Env-aligned metadata stores feature columns as feature_cols.
    Older dataset metadata may store state_columns.
    """
    for key in ["feature_cols", "state_columns", "state_cols"]:
        cols = metadata.get(key, None)
        if isinstance(cols, list) and len(cols) > 0:
            return [str(c) for c in cols]
    return None


def find_sofa_index(metadata: Dict, state_dim: int) -> Optional[int]:
    feature_cols = get_feature_columns_from_metadata(metadata)

    if feature_cols is None:
        # Fallback for the current SepsisTrajectoryEnv feature order:
        # [anchor_age, gender_encoded, heart_rate, mean_arterial_pressure,
        #  respiratory_rate, spo2, temperature, lactate, creatinine, wbc,
        #  platelets, bilirubin, sofa_proxy, fluid_amount, vasopressor_rate]
        if state_dim == 15:
            return 12
        return None

    possible_names = ["sofa_proxy", "sofa", "sofa_like", "severity", "severity_proxy"]
    lower_cols = [str(c).lower() for c in feature_cols]

    for name in possible_names:
        if name in lower_cols:
            idx = lower_cols.index(name)
            if 0 <= idx < state_dim:
                return idx

    return None


def severity_subgroup_analysis(
    states: np.ndarray,
    dataset_actions: np.ndarray,
    greedy_actions: np.ndarray,
    rewards: np.ndarray,
    metadata: Dict,
) -> Optional[pd.DataFrame]:
    sofa_idx = find_sofa_index(metadata, state_dim=states.shape[1])
    if sofa_idx is None:
        return None

    severity_values = states[:, sofa_idx].astype(float)
    if not np.isfinite(severity_values).all():
        return None

    # Since states are usually normalized for CQL, quantile groups are safer than
    # hard clinical thresholds at evaluation time.
    try:
        labels = pd.qcut(
            severity_values,
            q=3,
            labels=["Low", "Moderate", "High"],
            duplicates="drop",
        )
    except ValueError:
        return None

    df = pd.DataFrame({
        "severity_group": labels.astype(str),
        "severity_value": severity_values,
        "dataset_action": dataset_actions,
        "cql_greedy_action": greedy_actions,
        "reward": rewards,
    })

    df["dataset_fluid_bin"] = df["dataset_action"] // 5
    df["dataset_vaso_bin"] = df["dataset_action"] % 5
    df["cql_fluid_bin"] = df["cql_greedy_action"] // 5
    df["cql_vaso_bin"] = df["cql_greedy_action"] % 5

    order = ["Low", "Moderate", "High"]
    rows = []
    for group in order:
        g = df[df["severity_group"] == group]
        if g.empty:
            continue
        rows.append({
            "severity_group": group,
            "n_transitions": int(len(g)),
            "severity_mean_normalized": float(g["severity_value"].mean()),
            "dataset_mean_fluid_bin": float(g["dataset_fluid_bin"].mean()),
            "dataset_mean_vaso_bin": float(g["dataset_vaso_bin"].mean()),
            "cql_mean_fluid_bin": float(g["cql_fluid_bin"].mean()),
            "cql_mean_vaso_bin": float(g["cql_vaso_bin"].mean()),
            "mean_reward": float(g["reward"].mean()),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_action_distribution(action_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    x = np.arange(len(action_df))
    width = 0.4
    plt.bar(x - width / 2, action_df["dataset_percent"], width=width, label="Dataset / clinician actions")
    plt.bar(x + width / 2, action_df["cql_greedy_percent"], width=width, label="CQL greedy actions")
    plt.xlabel("Action index")
    plt.ylabel("Proportion")
    plt.title("Dataset vs CQL Greedy Action Distribution")
    plt.xticks(x, action_df["action"].astype(str), rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cql_vs_dataset_action_distribution.png", dpi=300)
    plt.close()


def plot_action_heatmap(action_heatmap_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = action_heatmap_df.pivot(index="fluid_bin", columns="vaso_bin", values="percent")
    pivot = pivot.reindex(index=range(5), columns=range(5), fill_value=0.0)

    plt.figure(figsize=(6, 5))
    image = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(image, label="CQL greedy action proportion")
    plt.xlabel("Vasopressor intensity bin")
    plt.ylabel("Fluid intensity bin")
    plt.title("CQL Greedy Policy Heatmap")
    plt.xticks(range(5), range(5))
    plt.yticks(range(5), range(5))

    for i in range(5):
        for j in range(5):
            value = pivot.values[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "cql_action_heatmap.png", dpi=300)
    plt.close()


def plot_severity_vasopressor(severity_df: pd.DataFrame, output_dir: Path) -> None:
    if severity_df is None or severity_df.empty:
        return

    plt.figure(figsize=(7, 5))
    x = np.arange(len(severity_df))
    width = 0.35
    plt.bar(x - width / 2, severity_df["dataset_mean_vaso_bin"], width=width, label="Dataset")
    plt.bar(x + width / 2, severity_df["cql_mean_vaso_bin"], width=width, label="CQL greedy")
    plt.xlabel("Severity subgroup")
    plt.ylabel("Mean vasopressor intensity bin")
    plt.title("CQL Vasopressor Intensity Across Severity Subgroups")
    plt.xticks(x, severity_df["severity_group"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cql_severity_vasopressor.png", dpi=300)
    plt.close()


def plot_q_value_histogram(max_q_values: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(max_q_values, bins=40)
    plt.xlabel("Max Q-value across actions")
    plt.ylabel("Frequency")
    plt.title("Distribution of CQL Estimated State Values")
    plt.tight_layout()
    plt.savefig(output_dir / "cql_q_value_histogram.png", dpi=300)
    plt.close()


def plot_td_error_histogram(td_error: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(td_error, bins=40)
    plt.xlabel("TD error")
    plt.ylabel("Frequency")
    plt.title("CQL Bellman TD Error Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "cql_td_error_histogram.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate(args) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    dataset_path = Path(args.dataset_path)
    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path) if args.metadata_path else None

    dataset = load_npz_dataset(dataset_path)
    dataset = sanitize_dataset(dataset, num_actions=args.num_actions)
    metadata = load_metadata(metadata_path)

    states = dataset["states"]
    actions = dataset["actions"]
    rewards = dataset["rewards"].astype(np.float32)
    next_states = dataset["next_states"]
    dones = dataset["dones"].astype(np.float32)

    state_dim = states.shape[1]
    num_transitions = states.shape[0]

    q_net, target_net, checkpoint = load_model(
        model_path=model_path,
        state_dim=state_dim,
        num_actions=args.num_actions,
        device=device,
        fallback_hidden_sizes=args.hidden_sizes,
    )

    num_actions = int(checkpoint.get("num_actions", args.num_actions))

    print(f"[INFO] Loaded dataset: {dataset_path}")
    print(f"[INFO] Loaded model: {model_path}")
    print(f"[INFO] Loaded metadata: {metadata_path}")
    print(f"[INFO] Number of transitions: {num_transitions}")
    print(f"[INFO] State dimension: {state_dim}")
    print(f"[INFO] Number of actions: {num_actions}")

    q_values = batched_q_values(q_net, states, device=device, batch_size=args.batch_size)
    next_q_values = batched_q_values(
        target_net if target_net is not None else q_net,
        next_states,
        device=device,
        batch_size=args.batch_size,
    )

    greedy_actions = np.argmax(q_values, axis=1).astype(np.int64)
    max_q_values = np.max(q_values, axis=1)
    q_dataset_actions = q_values[np.arange(len(actions)), actions]
    next_max_q = np.max(next_q_values, axis=1)

    td_error, td_metrics = compute_td_diagnostics(
        q_dataset_actions=q_dataset_actions,
        rewards=rewards,
        dones=dones,
        next_max_q=next_max_q,
        gamma=args.gamma,
    )

    dataset_fluid = action_to_fluid(actions)
    dataset_vaso = action_to_vaso(actions)
    greedy_fluid = action_to_fluid(greedy_actions)
    greedy_vaso = action_to_vaso(greedy_actions)

    dataset_counts = np.bincount(actions, minlength=num_actions)
    greedy_counts = np.bincount(greedy_actions, minlength=num_actions)
    dataset_probs = dataset_counts / max(dataset_counts.sum(), 1)
    greedy_probs = greedy_counts / max(greedy_counts.sum(), 1)

    action_match_rate = float(np.mean(greedy_actions == actions))

    metrics = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "source_environment": metadata.get("source_environment", "unknown"),
        "dataset_type": metadata.get("dataset_type", "unknown"),
        "feature_cols": get_feature_columns_from_metadata(metadata),
        "normalization_used": metadata.get("normalization_used", None),
        "num_transitions": int(num_transitions),
        "state_dim": int(state_dim),
        "num_actions": int(num_actions),
        "gamma": float(args.gamma),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", math.nan)),
        "dataset_reward_mean": float(np.mean(rewards)),
        "dataset_reward_std": float(np.std(rewards)),
        "dataset_reward_min": float(np.min(rewards)),
        "dataset_reward_max": float(np.max(rewards)),
        "dataset_done_rate": float(np.mean(dones)),
        "dataset_mean_fluid_bin": float(np.mean(dataset_fluid)),
        "dataset_mean_vaso_bin": float(np.mean(dataset_vaso)),
        "cql_greedy_mean_fluid_bin": float(np.mean(greedy_fluid)),
        "cql_greedy_mean_vaso_bin": float(np.mean(greedy_vaso)),
        "cql_greedy_action_entropy": safe_entropy(greedy_probs),
        "dataset_action_entropy": safe_entropy(dataset_probs),
        "cql_dataset_action_match_rate": action_match_rate,
        "q_dataset_action_mean": float(np.mean(q_dataset_actions)),
        "q_dataset_action_std": float(np.std(q_dataset_actions)),
        "q_max_mean": float(np.mean(max_q_values)),
        "q_max_std": float(np.std(max_q_values)),
        "q_max_min": float(np.min(max_q_values)),
        "q_max_median": float(np.median(max_q_values)),
        "q_max_max": float(np.max(max_q_values)),
        **td_metrics,
        **estimate_observed_dataset_episode_returns(dataset),
        **estimate_direct_method_value(dataset, max_q_values),
    }

    # Reports
    action_df = make_action_distribution_df(
        dataset_actions=actions,
        greedy_actions=greedy_actions,
        num_actions=num_actions,
    )
    action_heatmap_df = make_action_heatmap_df(greedy_actions=greedy_actions, num_actions=num_actions)
    q_summary_df = make_q_value_summary_df(q_values)
    severity_df = severity_subgroup_analysis(
        states=states,
        dataset_actions=actions,
        greedy_actions=greedy_actions,
        rewards=rewards,
        metadata=metadata,
    )

    transition_df = pd.DataFrame({
        "dataset_action": actions.astype(int),
        "dataset_fluid_bin": dataset_fluid.astype(int),
        "dataset_vaso_bin": dataset_vaso.astype(int),
        "cql_greedy_action": greedy_actions.astype(int),
        "cql_fluid_bin": greedy_fluid.astype(int),
        "cql_vaso_bin": greedy_vaso.astype(int),
        "reward": rewards.astype(float),
        "done": dones.astype(int),
        "q_dataset_action": q_dataset_actions.astype(float),
        "q_cql_greedy_action": max_q_values.astype(float),
        "td_error": td_error.astype(float),
    })

    if "stay_ids" in dataset:
        transition_df.insert(0, "stay_id", dataset["stay_ids"])
    if "timesteps" in dataset:
        insert_at = 1 if "stay_ids" in dataset else 0
        transition_df.insert(insert_at, "timestep", dataset["timesteps"].astype(int))

    metrics_path = output_dir / "cql_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    action_csv = output_dir / "cql_eval_action_distribution.csv"
    heatmap_csv = output_dir / "cql_action_heatmap.csv"
    q_summary_csv = output_dir / "cql_q_value_summary.csv"
    transition_csv = output_dir / "cql_transition_predictions.csv"

    action_df.to_csv(action_csv, index=False)
    action_heatmap_df.to_csv(heatmap_csv, index=False)
    q_summary_df.to_csv(q_summary_csv, index=False)
    transition_df.to_csv(transition_csv, index=False)

    if severity_df is not None:
        severity_csv = output_dir / "cql_severity_subgroup_policy.csv"
        severity_df.to_csv(severity_csv, index=False)
    else:
        severity_csv = None
        print("[WARN] Severity subgroup analysis skipped because sofa_proxy metadata was not available.")

    # Plots
    plot_action_distribution(action_df, output_dir)
    plot_action_heatmap(action_heatmap_df, output_dir)
    plot_q_value_histogram(max_q_values, output_dir)
    plot_td_error_histogram(td_error, output_dir)
    if severity_df is not None:
        plot_severity_vasopressor(severity_df, output_dir)

    print("\n[DONE] CQL evaluation completed successfully.")
    print(f"CQL greedy mean fluid bin: {metrics['cql_greedy_mean_fluid_bin']:.3f}")
    print(f"CQL greedy mean vasopressor bin: {metrics['cql_greedy_mean_vaso_bin']:.3f}")
    print(f"Dataset mean fluid bin: {metrics['dataset_mean_fluid_bin']:.3f}")
    print(f"Dataset mean vasopressor bin: {metrics['dataset_mean_vaso_bin']:.3f}")
    print(f"Action match rate: {metrics['cql_dataset_action_match_rate']:.3f}")
    print(f"DM estimated CQL initial state value mean: {metrics['dm_estimated_cql_initial_state_value_mean']:.4f}")
    print(f"Bellman MSE: {metrics['bellman_mse']:.4f}")

    print("\nOutput files:")
    print(f"  - {metrics_path}")
    print(f"  - {action_csv}")
    print(f"  - {heatmap_csv}")
    print(f"  - {q_summary_csv}")
    print(f"  - {transition_csv}")
    if severity_csv is not None:
        print(f"  - {severity_csv}")
    print(f"  - {output_dir / 'cql_vs_dataset_action_distribution.png'}")
    print(f"  - {output_dir / 'cql_action_heatmap.png'}")
    print(f"  - {output_dir / 'cql_q_value_histogram.png'}")
    print(f"  - {output_dir / 'cql_td_error_histogram.png'}")
    if severity_df is not None:
        print(f"  - {output_dir / 'cql_severity_vasopressor.png'}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained discrete CQL agent on an env-aligned offline sepsis RL dataset."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="offline_dataset_env_aligned/offline_cql_dataset.npz",
        help="Path to offline_cql_dataset.npz created by prepare_offline_dataset_env_aligned.py.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="cql_results_env_aligned/cql_q_network.pt",
        help="Path to trained CQL checkpoint created by train_cql_agent.py.",
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        default="offline_dataset_env_aligned/offline_cql_metadata.json",
        help="Metadata JSON created by prepare_offline_dataset_env_aligned.py.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cql_evaluation_results_env_aligned",
        help="Directory where evaluation outputs will be saved.",
    )

    parser.add_argument("--num_actions", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--hidden_sizes", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, cuda:0, etc. Default auto-detects.")

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
