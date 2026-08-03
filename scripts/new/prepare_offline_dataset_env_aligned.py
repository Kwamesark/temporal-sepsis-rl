#!/usr/bin/env python3
"""
prepare_offline_dataset_env_aligned.py

Create an offline CQL dataset that corresponds directly to the same temporal
sepsis environment used for PPO and A2C.

This script uses SepsisTrajectoryEnv as the single source of truth for:
    - state/features
    - 25-action fluid-vasopressor action space
    - temporal episode structure
    - reward function

Expected project structure:
    project/
    ├── sepsis_temporal_env.py
    ├── prepare_offline_dataset_env_aligned.py
    └── data/
        └── sepsis_trajectories.csv

Input CSV must match the columns required by SepsisTrajectoryEnv.

Output:
    offline_dataset_env_aligned/offline_cql_dataset.npz
    offline_dataset_env_aligned/offline_cql_transitions.csv
    offline_dataset_env_aligned/offline_cql_metadata.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sepsis_temporal_env import SepsisTrajectoryEnv


def zscore_normalize(
    states: np.ndarray,
    next_states: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, list]]:
    """
    Normalize states using statistics estimated from current states.
    The same statistics are applied to next_states.
    """
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std = np.where(std < eps, 1.0, std)

    states_norm = (states - mean) / std
    next_states_norm = (next_states - mean) / std

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    return states_norm.astype(np.float32), next_states_norm.astype(np.float32), stats


def build_env_aligned_transitions(
    env: SepsisTrajectoryEnv,
    normalize: bool = True,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict]:
    """
    Build offline transitions from SepsisTrajectoryEnv trajectories.

    Offline CQL uses the observed clinician action at each 4-hour window:
        action_t = current_row["action"]

    The reward is computed using env._get_reward(action_t, current_row, next_row),
    so it matches the PPO/A2C environment reward exactly.
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    stay_ids = []
    timesteps = []

    transition_rows = []
    episode_lengths = []

    for stay_id, traj in env.trajectories.items():
        traj = traj.sort_values("timestep").reset_index(drop=True)

        if len(traj) < 2:
            continue

        episode_lengths.append(len(traj))

        # Set environment trajectory context so env._get_reward terminal logic matches PPO/A2C.
        env.current_stay_id = stay_id
        env.current_traj = traj

        for i in range(len(traj) - 1):
            env.current_timestep = i

            current_row = traj.iloc[i]
            next_row = traj.iloc[i + 1]

            state = current_row[env.feature_cols].astype(float).to_numpy(dtype=np.float32)
            next_state = next_row[env.feature_cols].astype(float).to_numpy(dtype=np.float32)

            # Offline RL should train on observed clinician actions.
            action = int(current_row["action"])
            action = max(0, min(24, action))

            reward = float(env._get_reward(action, current_row, next_row))
            done = bool(i == len(traj) - 2)

            fluid_bin = int(current_row["fluid_bin"])
            vasopressor_bin = int(current_row["vasopressor_bin"])

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            stay_ids.append(stay_id)
            timesteps.append(int(current_row["timestep"]))

            transition_rows.append({
                "stay_id": stay_id,
                "timestep": int(current_row["timestep"]),
                "action": action,
                "fluid_bin": fluid_bin,
                "vasopressor_bin": vasopressor_bin,
                "reward": reward,
                "done": int(done),
                "lactate": float(current_row["lactate"]),
                "next_lactate": float(next_row["lactate"]),
                "mean_arterial_pressure": float(current_row["mean_arterial_pressure"]),
                "next_mean_arterial_pressure": float(next_row["mean_arterial_pressure"]),
                "sofa_proxy": float(current_row["sofa_proxy"]),
                "next_sofa_proxy": float(next_row["sofa_proxy"]),
                "hospital_expire_flag": int(current_row["hospital_expire_flag"]),
            })

    if len(states) == 0:
        raise ValueError("No transitions were created. Check that trajectories have at least two timesteps.")

    states = np.asarray(states, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)
    normalization_stats = None

    if normalize:
        states, next_states, normalization_stats = zscore_normalize(states, next_states)

    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)
    stay_ids = np.asarray(stay_ids)
    timesteps = np.asarray(timesteps, dtype=np.int64)

    transitions_df = pd.DataFrame(transition_rows)

    metadata = {
        "dataset_type": "env_aligned_offline_cql_dataset",
        "source_environment": "SepsisTrajectoryEnv",
        "num_transitions": int(len(states)),
        "num_episodes": int(len(episode_lengths)),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "min_episode_length": int(np.min(episode_lengths)) if episode_lengths else 0,
        "max_episode_length": int(np.max(episode_lengths)) if episode_lengths else 0,
        "state_dim": int(states.shape[1]),
        "num_actions": int(env.action_space.n),
        "feature_cols": list(env.feature_cols),
        "action_formula": "action = fluid_bin * 5 + vasopressor_bin",
        "offline_action_source": "observed clinician action from trajectory CSV",
        "reward_source": "SepsisTrajectoryEnv._get_reward using clinician action",
        "normalization_used": bool(normalize),
        "normalization_stats": normalization_stats,
        "reward_components": [
            "lactate improvement",
            "MAP stabilization",
            "SOFA proxy reduction",
            "severity-aware treatment logic",
            "avoid overly aggressive low-risk treatment",
            "behavior-cloning-style action distance penalty",
            "terminal survival/mortality reward",
        ],
    }

    dataset = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "stay_ids": stay_ids,
        "timesteps": timesteps,
    }

    return dataset, transitions_df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an env-aligned offline CQL dataset for sepsis temporal RL."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sepsis_trajectories.csv",
        help="Path to the same trajectory CSV used by SepsisTrajectoryEnv for PPO/A2C.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="offline_dataset_env_aligned",
        help="Directory where the CQL offline dataset will be saved.",
    )

    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable z-score normalization of CQL states.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading SepsisTrajectoryEnv from: {args.data_path}")
    env = SepsisTrajectoryEnv(data_path=args.data_path)

    print("[INFO] Building offline transitions aligned with PPO/A2C environment...")
    dataset, transitions_df, metadata = build_env_aligned_transitions(
        env=env,
        normalize=not args.no_normalize,
    )

    npz_path = output_dir / "offline_cql_dataset.npz"
    csv_path = output_dir / "offline_cql_transitions.csv"
    metadata_path = output_dir / "offline_cql_metadata.json"

    np.savez_compressed(
        npz_path,
        states=dataset["states"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        next_states=dataset["next_states"],
        dones=dataset["dones"],
        stay_ids=dataset["stay_ids"],
        timesteps=dataset["timesteps"],
    )

    transitions_df.to_csv(csv_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n[DONE] Env-aligned offline CQL dataset prepared successfully.")
    print(f"Transitions: {metadata['num_transitions']}")
    print(f"Episodes: {metadata['num_episodes']}")
    print(f"State dimension: {metadata['state_dim']}")
    print(f"Action space: {metadata['num_actions']}")
    print(f"Mean episode length: {metadata['mean_episode_length']:.2f}")
    print(f"Normalization used: {metadata['normalization_used']}")
    print("\nOutput files:")
    print(f"  - {npz_path}")
    print(f"  - {csv_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()
