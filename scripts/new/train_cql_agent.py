#!/usr/bin/env python3
"""
train_cql_agent.py

Train a discrete Conservative Q-Learning (CQL) baseline for offline sepsis
reinforcement learning using transition tuples created by
prepare_offline_dataset_env_aligned.py.

This version is aligned with SepsisTrajectoryEnv, which is also used for PPO/A2C:
    - same state feature dimension
    - same 25-action fluid-vasopressor treatment space
    - same reward values already computed during offline dataset preparation
    - same trajectory identifiers for episode-level train/validation splitting

Expected input NPZ file:
    offline_dataset_env_aligned/offline_cql_dataset.npz

Expected arrays inside NPZ:
    states       : shape [N, state_dim]
    actions      : shape [N]
    rewards      : shape [N]
    next_states  : shape [N, state_dim]
    dones        : shape [N]

Optional arrays:
    stay_ids     : shape [N]
    timesteps    : shape [N]

The model uses a DQN-style Bellman objective plus the discrete CQL penalty:

    CQL loss = logsumexp(Q(s, all_actions)) - Q(s, dataset_action)
    Total loss = Bellman loss + alpha * CQL loss

Outputs:
    cql_q_network.pt
    cql_training_log.csv
    cql_metrics.json
    cql_action_distribution.csv
    cql_loss_curve.png
    cql_action_distribution.png
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep benchmark enabled for speed. Change to deterministic=True only if
    # exact CUDA reproducibility is required.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


class OfflineTransitionDataset(Dataset):
    """PyTorch dataset for offline RL transition tuples."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)
        self.next_states = torch.as_tensor(next_states, dtype=torch.float32)
        self.dones = torch.as_tensor(dones, dtype=torch.float32)

        if self.states.ndim != 2:
            raise ValueError(f"states must be 2D, got shape {self.states.shape}")
        if self.next_states.shape != self.states.shape:
            raise ValueError(
                f"next_states shape {self.next_states.shape} must match states shape {self.states.shape}"
            )
        if len(self.actions) != len(self.states):
            raise ValueError("actions length must match number of states")
        if len(self.rewards) != len(self.states):
            raise ValueError("rewards length must match number of states")
        if len(self.dones) != len(self.states):
            raise ValueError("dones length must match number of states")

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class QNetwork(nn.Module):
    """MLP Q-network for discrete-action CQL."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 25,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
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
# Data loading and cleaning
# ---------------------------------------------------------------------


def load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load offline CQL dataset from NPZ."""
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
    """Remove invalid rows and check action bounds."""
    states = dataset["states"]
    next_states = dataset["next_states"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    dones = dataset["dones"]

    finite_mask = (
        np.isfinite(states).all(axis=1)
        & np.isfinite(next_states).all(axis=1)
        & np.isfinite(rewards)
        & np.isfinite(dones)
        & np.isfinite(actions)
    )

    action_mask = (actions >= 0) & (actions < num_actions)
    keep_mask = finite_mask & action_mask

    removed = int(len(actions) - keep_mask.sum())
    if removed > 0:
        print(f"[WARN] Removed {removed} invalid transitions.")

    cleaned: Dict[str, np.ndarray] = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and len(value) == len(actions):
            cleaned[key] = value[keep_mask]
        else:
            cleaned[key] = value

    if len(cleaned["actions"]) == 0:
        raise ValueError("No valid transitions remain after cleaning.")

    return cleaned


# ---------------------------------------------------------------------
# Train/validation splitting
# ---------------------------------------------------------------------


def split_indices_by_episode(
    stay_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Create an episode-level train/validation split.

    This prevents timesteps from the same ICU stay from appearing in both
    training and validation sets.
    """
    rng = np.random.default_rng(seed)
    unique_stays = np.unique(stay_ids)

    if len(unique_stays) < 2:
        raise ValueError("Episode-level split requires at least two unique stay_ids.")

    shuffled_stays = unique_stays.copy()
    rng.shuffle(shuffled_stays)

    val_episode_count = int(round(len(shuffled_stays) * val_fraction))
    val_episode_count = max(1, min(val_episode_count, len(shuffled_stays) - 1))

    val_stays = set(shuffled_stays[:val_episode_count])
    train_stays = set(shuffled_stays[val_episode_count:])

    train_indices = np.asarray(
        [idx for idx, stay_id in enumerate(stay_ids) if stay_id in train_stays],
        dtype=np.int64,
    )
    val_indices = np.asarray(
        [idx for idx, stay_id in enumerate(stay_ids) if stay_id in val_stays],
        dtype=np.int64,
    )

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Episode-level split produced an empty train or validation set.")

    split_info: Dict[str, object] = {
        "split_method": "episode_level_by_stay_id",
        "train_episodes": int(len(train_stays)),
        "val_episodes": int(len(val_stays)),
        "train_transitions": int(len(train_indices)),
        "val_transitions": int(len(val_indices)),
    }

    return train_indices, val_indices, split_info


def create_train_val_subsets(
    full_dataset: OfflineTransitionDataset,
    raw_dataset: Dict[str, np.ndarray],
    val_fraction: float,
    seed: int,
    transition_split: bool,
) -> Tuple[Subset, Subset, Dict[str, object]]:
    """Create train/validation subsets, preferring episode-level split."""
    if (not transition_split) and "stay_ids" in raw_dataset:
        try:
            stay_ids = raw_dataset["stay_ids"]
            if len(stay_ids) == len(full_dataset):
                train_indices, val_indices, split_info = split_indices_by_episode(
                    stay_ids=stay_ids,
                    val_fraction=val_fraction,
                    seed=seed,
                )

                train_subset = Subset(full_dataset, train_indices.tolist())
                val_subset = Subset(full_dataset, val_indices.tolist())
                return train_subset, val_subset, split_info

            print("[WARN] stay_ids length does not match dataset length. Falling back to transition split.")
        except Exception as exc:
            print(f"[WARN] Episode-level split failed: {exc}")
            print("[WARN] Falling back to transition-level split.")

    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError("Train/validation split failed. Adjust --val_fraction.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    split_info = {
        "split_method": "transition_level_random_split",
        "train_episodes": None,
        "val_episodes": None,
        "train_transitions": int(train_size),
        "val_transitions": int(val_size),
    }

    return train_subset, val_subset, split_info


# ---------------------------------------------------------------------
# CQL loss and model evaluation
# ---------------------------------------------------------------------


def compute_cql_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    batch,
    gamma: float,
    cql_alpha: float,
    device: torch.device,
    double_q: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    states, actions, rewards, next_states, dones = batch

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    q_values = q_net(states)
    q_dataset_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double_q:
            next_actions = torch.argmax(q_net(next_states), dim=1, keepdim=True)
            next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = target_net(next_states).max(dim=1).values

        target_q = rewards + gamma * (1.0 - dones) * next_q_values

    bellman_loss = F.mse_loss(q_dataset_actions, target_q)

    # Discrete CQL regularizer. It penalizes assigning overly high Q-values
    # to actions that are not supported by the offline clinician dataset.
    cql_penalty = torch.logsumexp(q_values, dim=1).mean() - q_dataset_actions.mean()

    total_loss = bellman_loss + cql_alpha * cql_penalty

    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "bellman_loss": float(bellman_loss.detach().cpu().item()),
        "cql_penalty": float(cql_penalty.detach().cpu().item()),
        "q_dataset_mean": float(q_dataset_actions.detach().mean().cpu().item()),
        "target_q_mean": float(target_q.detach().mean().cpu().item()),
    }

    return total_loss, metrics


@torch.no_grad()
def evaluate_model(
    q_net: QNetwork,
    target_net: QNetwork,
    loader: DataLoader,
    gamma: float,
    cql_alpha: float,
    device: torch.device,
    num_actions: int,
) -> Dict[str, float]:
    q_net.eval()

    losses = []
    bellman_losses = []
    cql_penalties = []
    q_dataset_means = []
    target_q_means = []
    greedy_actions_all = []
    dataset_actions_all = []
    rewards_all = []

    for batch in loader:
        _, batch_metrics = compute_cql_loss(
            q_net=q_net,
            target_net=target_net,
            batch=batch,
            gamma=gamma,
            cql_alpha=cql_alpha,
            device=device,
            double_q=True,
        )

        states, actions, rewards, _, _ = batch
        states = states.to(device)

        q_values = q_net(states)
        greedy_actions = torch.argmax(q_values, dim=1).detach().cpu().numpy()

        greedy_actions_all.append(greedy_actions)
        dataset_actions_all.append(actions.numpy())
        rewards_all.append(rewards.numpy())

        losses.append(batch_metrics["loss"])
        bellman_losses.append(batch_metrics["bellman_loss"])
        cql_penalties.append(batch_metrics["cql_penalty"])
        q_dataset_means.append(batch_metrics["q_dataset_mean"])
        target_q_means.append(batch_metrics["target_q_mean"])

    greedy_actions_all = np.concatenate(greedy_actions_all)
    dataset_actions_all = np.concatenate(dataset_actions_all)
    rewards_all = np.concatenate(rewards_all)

    greedy_counts = np.bincount(greedy_actions_all, minlength=num_actions)
    dataset_counts = np.bincount(dataset_actions_all, minlength=num_actions)

    greedy_probs = greedy_counts / max(greedy_counts.sum(), 1)
    dataset_probs = dataset_counts / max(dataset_counts.sum(), 1)

    greedy_fluid = greedy_actions_all // 5
    greedy_vaso = greedy_actions_all % 5
    dataset_fluid = dataset_actions_all // 5
    dataset_vaso = dataset_actions_all % 5

    metrics = {
        "val_loss": float(np.mean(losses)),
        "val_bellman_loss": float(np.mean(bellman_losses)),
        "val_cql_penalty": float(np.mean(cql_penalties)),
        "val_q_dataset_mean": float(np.mean(q_dataset_means)),
        "val_target_q_mean": float(np.mean(target_q_means)),
        "val_mean_observed_dataset_reward": float(np.mean(rewards_all)),
        "greedy_mean_fluid_bin": float(np.mean(greedy_fluid)),
        "greedy_mean_vaso_bin": float(np.mean(greedy_vaso)),
        "dataset_mean_fluid_bin": float(np.mean(dataset_fluid)),
        "dataset_mean_vaso_bin": float(np.mean(dataset_vaso)),
        "greedy_action_entropy": float(-(greedy_probs * np.log(greedy_probs + 1e-12)).sum()),
        "dataset_action_entropy": float(-(dataset_probs * np.log(dataset_probs + 1e-12)).sum()),
    }

    return metrics


# ---------------------------------------------------------------------
# Target-network updates
# ---------------------------------------------------------------------


def hard_update(target_net: nn.Module, source_net: nn.Module) -> None:
    target_net.load_state_dict(source_net.state_dict())


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


@torch.no_grad()
def create_action_distribution_report(
    q_net: QNetwork,
    states: np.ndarray,
    dataset_actions: np.ndarray,
    device: torch.device,
    num_actions: int,
    output_dir: Path,
) -> pd.DataFrame:
    q_net.eval()

    batch_size = 4096
    greedy_actions = []

    for start in range(0, len(states), batch_size):
        end = start + batch_size
        batch_states = torch.as_tensor(states[start:end], dtype=torch.float32).to(device)
        q_values = q_net(batch_states)
        batch_actions = torch.argmax(q_values, dim=1).detach().cpu().numpy()
        greedy_actions.append(batch_actions)

    greedy_actions = np.concatenate(greedy_actions)

    dataset_counts = np.bincount(dataset_actions, minlength=num_actions)
    greedy_counts = np.bincount(greedy_actions, minlength=num_actions)

    rows = []
    for action in range(num_actions):
        rows.append(
            {
                "action": action,
                "fluid_bin": action // 5,
                "vasopressor_bin": action % 5,
                "dataset_count": int(dataset_counts[action]),
                "dataset_percent": float(dataset_counts[action] / max(dataset_counts.sum(), 1)),
                "cql_greedy_count": int(greedy_counts[action]),
                "cql_greedy_percent": float(greedy_counts[action] / max(greedy_counts.sum(), 1)),
            }
        )

    action_df = pd.DataFrame(rows)
    action_csv = output_dir / "cql_action_distribution.csv"
    action_df.to_csv(action_csv, index=False)

    return action_df


def estimate_observed_clinician_episode_returns(dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Summarize observed offline returns grouped by stay_id.

    Important: these are clinician-behavior returns from the offline dataset,
    not learned CQL policy returns.
    """
    if "stay_ids" not in dataset:
        return {}

    rewards = dataset["rewards"].astype(float)
    stay_ids = dataset["stay_ids"]

    if len(stay_ids) != len(rewards):
        return {}

    df = pd.DataFrame({"stay_id": stay_ids, "reward": rewards})
    returns = df.groupby("stay_id")["reward"].sum().values

    if len(returns) == 0:
        return {}

    return {
        "observed_clinician_episode_return_mean": float(np.mean(returns)),
        "observed_clinician_episode_return_std": float(np.std(returns)),
        "observed_clinician_episode_return_min": float(np.min(returns)),
        "observed_clinician_episode_return_max": float(np.max(returns)),
        "observed_clinician_num_episodes": int(len(returns)),
    }


def plot_loss_curve(log_df: pd.DataFrame, output_dir: Path) -> None:
    if log_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CQL Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cql_loss_curve.png", dpi=300)
    plt.close()


def plot_action_distribution(action_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    x = np.arange(len(action_df))
    width = 0.4
    plt.bar(x - width / 2, action_df["dataset_percent"], width=width, label="Dataset actions")
    plt.bar(x + width / 2, action_df["cql_greedy_percent"], width=width, label="CQL greedy actions")
    plt.xlabel("Action index")
    plt.ylabel("Proportion")
    plt.title("Dataset vs CQL Greedy Action Distribution")
    plt.xticks(x, action_df["action"].astype(str), rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cql_action_distribution.png", dpi=300)
    plt.close()


def save_config(args, output_dir: Path) -> None:
    config_path = output_dir / "cql_training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------


def train(args) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, output_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    dataset_path = Path(args.dataset_path)
    dataset = load_npz_dataset(dataset_path)
    dataset = sanitize_dataset(dataset, num_actions=args.num_actions)

    states = dataset["states"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_states = dataset["next_states"]
    dones = dataset["dones"]

    state_dim = states.shape[1]
    num_transitions = states.shape[0]

    print(f"[INFO] Loaded dataset: {dataset_path}")
    print(f"[INFO] Number of transitions: {num_transitions}")
    print(f"[INFO] State dimension: {state_dim}")
    print(f"[INFO] Number of actions: {args.num_actions}")
    print(f"[INFO] Reward mean/std: {np.mean(rewards):.4f} / {np.std(rewards):.4f}")

    full_dataset = OfflineTransitionDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    train_dataset, val_dataset, split_info = create_train_val_subsets(
        full_dataset=full_dataset,
        raw_dataset=dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        transition_split=args.transition_split,
    )

    print(f"[INFO] Split method: {split_info['split_method']}")
    print(f"[INFO] Train transitions: {split_info['train_transitions']}")
    print(f"[INFO] Validation transitions: {split_info['val_transitions']}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())

    q_net = QNetwork(
        state_dim=state_dim,
        num_actions=args.num_actions,
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
    ).to(device)

    target_net = QNetwork(
        state_dim=state_dim,
        num_actions=args.num_actions,
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
    ).to(device)

    hard_update(target_net, q_net)
    target_net.eval()

    optimizer = torch.optim.AdamW(
        q_net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_model_path = output_dir / "cql_q_network.pt"
    log_rows = []

    print("[INFO] Starting CQL training...")

    for epoch in range(1, args.epochs + 1):
        q_net.train()

        train_losses = []
        train_bellman_losses = []
        train_cql_penalties = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            loss, batch_metrics = compute_cql_loss(
                q_net=q_net,
                target_net=target_net,
                batch=batch,
                gamma=args.gamma,
                cql_alpha=args.cql_alpha,
                device=device,
                double_q=not args.no_double_q,
            )

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)

            optimizer.step()

            if args.soft_target_update:
                soft_update(target_net, q_net, tau=args.tau)

            train_losses.append(batch_metrics["loss"])
            train_bellman_losses.append(batch_metrics["bellman_loss"])
            train_cql_penalties.append(batch_metrics["cql_penalty"])

        if not args.soft_target_update and epoch % args.target_update_interval == 0:
            hard_update(target_net, q_net)

        val_metrics = evaluate_model(
            q_net=q_net,
            target_net=target_net,
            loader=val_loader,
            gamma=args.gamma,
            cql_alpha=args.cql_alpha,
            device=device,
            num_actions=args.num_actions,
        )

        train_loss = float(np.mean(train_losses))
        train_bellman_loss = float(np.mean(train_bellman_losses))
        train_cql_penalty = float(np.mean(train_cql_penalties))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bellman_loss": train_bellman_loss,
            "train_cql_penalty": train_cql_penalty,
            **val_metrics,
        }
        log_rows.append(row)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "model_state_dict": q_net.state_dict(),
                    "target_model_state_dict": target_net.state_dict(),
                    "state_dim": state_dim,
                    "num_actions": args.num_actions,
                    "hidden_sizes": hidden_sizes,
                    "dropout": args.dropout,
                    "args": vars(args),
                    "split_info": split_info,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                best_model_path,
            )

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"bellman={val_metrics['val_bellman_loss']:.4f} | "
                f"cql={val_metrics['val_cql_penalty']:.4f} | "
                f"greedy_fluid={val_metrics['greedy_mean_fluid_bin']:.2f} | "
                f"greedy_vaso={val_metrics['greedy_mean_vaso_bin']:.2f}"
            )

    print(f"[INFO] Best model saved to: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    q_net.load_state_dict(checkpoint["model_state_dict"])
    target_net.load_state_dict(checkpoint["target_model_state_dict"])

    final_val_metrics = evaluate_model(
        q_net=q_net,
        target_net=target_net,
        loader=val_loader,
        gamma=args.gamma,
        cql_alpha=args.cql_alpha,
        device=device,
        num_actions=args.num_actions,
    )

    action_df = create_action_distribution_report(
        q_net=q_net,
        states=states,
        dataset_actions=actions,
        device=device,
        num_actions=args.num_actions,
        output_dir=output_dir,
    )

    observed_return_metrics = estimate_observed_clinician_episode_returns(dataset)

    final_metrics = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "num_transitions": int(num_transitions),
        "state_dim": int(state_dim),
        "num_actions": int(args.num_actions),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "dataset_reward_mean": float(np.mean(rewards)),
        "dataset_reward_std": float(np.std(rewards)),
        "dataset_done_rate": float(np.mean(dones)),
        **split_info,
        **final_val_metrics,
        **observed_return_metrics,
    }

    metrics_path = output_dir / "cql_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    log_df = pd.DataFrame(log_rows)
    log_path = output_dir / "cql_training_log.csv"
    log_df.to_csv(log_path, index=False)

    plot_loss_curve(log_df, output_dir)
    plot_action_distribution(action_df, output_dir)

    print("\n[DONE] CQL training completed successfully.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best epoch: {final_metrics['best_epoch']}")
    print(f"CQL greedy mean fluid bin: {final_metrics['greedy_mean_fluid_bin']:.3f}")
    print(f"CQL greedy mean vasopressor bin: {final_metrics['greedy_mean_vaso_bin']:.3f}")
    print("\nOutput files:")
    print(f"  - {best_model_path}")
    print(f"  - {log_path}")
    print(f"  - {metrics_path}")
    print(f"  - {output_dir / 'cql_training_config.json'}")
    print(f"  - {output_dir / 'cql_action_distribution.csv'}")
    print(f"  - {output_dir / 'cql_loss_curve.png'}")
    print(f"  - {output_dir / 'cql_action_distribution.png'}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a discrete CQL baseline from an env-aligned offline sepsis RL dataset."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="offline_dataset_env_aligned/offline_cql_dataset.npz",
        help="Path to offline_cql_dataset.npz created by prepare_offline_dataset_env_aligned.py.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cql_results_env_aligned",
        help="Directory where CQL outputs will be saved.",
    )

    parser.add_argument("--num_actions", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cql_alpha", type=float, default=1.0)
    parser.add_argument("--hidden_sizes", type=str, default="256,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--target_update_interval", type=int, default=5)
    parser.add_argument("--soft_target_update", action="store_true")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--no_double_q", action="store_true")
    parser.add_argument("--transition_split", action="store_true", help="Use random transition-level split instead of episode-level split.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, cuda:0, etc. Default auto-detects.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
