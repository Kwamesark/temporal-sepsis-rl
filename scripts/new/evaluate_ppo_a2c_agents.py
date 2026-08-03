#!/usr/bin/env python3
"""
evaluate_ppo_a2c_agents.py

Evaluate trained PPO and A2C temporal 25-action policies using the same
SepsisTrajectoryEnv used for CQL dataset preparation.

This script:
    - Loads PPO and A2C trained models
    - Loads matching VecNormalize statistics
    - Evaluates deterministic policies
    - Uses raw environment rewards during evaluation
    - Saves episode-level rewards
    - Saves summary metrics
    - Saves action distribution results
    - Saves severity subgroup behavior
    - Generates evaluation plots

Aligned with:
    - sepsis_temporal_env.py
    - train_ppo_temporal_25.py
    - train_a2c_temporal_25.py
    - prepare_offline_dataset_env_aligned.py
    - train_cql_agent.py
    - evaluate_cql_agent.py
"""

import os
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sepsis_temporal_env import SepsisTrajectoryEnv


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ALGORITHMS = ["PPO", "A2C"]
SEEDS = [1,2,3, 4, 5]

PHASE = "temporal_25_action"
N_EPISODES = 50

DATA_PATH = "../data/sepsis_trajectories.csv"

MODEL_DIR = "../models"
RESULTS_DIR = "../results"
FIGURES_DIR = "../figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def decode_action(action: int):
    action = int(action)
    fluid_bin = action // 5
    vasopressor_bin = action % 5
    return fluid_bin, vasopressor_bin


def build_raw_env():
    return DummyVecEnv([
        lambda: SepsisTrajectoryEnv(data_path=DATA_PATH)
    ])


def load_model_and_env(algorithm: str, seed: int, device: str):
    algorithm_lower = algorithm.lower()

    model_path = f"{MODEL_DIR}/{algorithm_lower}_temporal_25_seed_{seed}"
    vecnormalize_path = f"{MODEL_DIR}/{algorithm_lower}_temporal_25_seed_{seed}_vecnormalize.pkl"

    if not Path(model_path + ".zip").exists():
        raise FileNotFoundError(f"Model file not found: {model_path}.zip")

    if not Path(vecnormalize_path).exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnormalize_path}")

    raw_env = build_raw_env()

    env = VecNormalize.load(
        vecnormalize_path,
        raw_env
    )

    # Critical for evaluation:
    # Do not update normalization statistics.
    # Do not normalize reward during evaluation.
    env.training = False
    env.norm_reward = False

    if algorithm == "PPO":
        model = PPO.load(
            model_path,
            env=env,
            device=device
        )
    elif algorithm == "A2C":
        model = A2C.load(
            model_path,
            env=env,
            device=device
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return model, env, model_path, vecnormalize_path


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_single_model(algorithm: str, seed: int, device: str):
    print(f"\nEvaluating {algorithm} temporal 25-action policy | Seed {seed}\n")

    set_seed(seed)

    model, env, model_path, vec_path = load_model_and_env(
        algorithm=algorithm,
        seed=seed,
        device=device
    )

    episode_rewards = []
    episode_lengths = []

    transition_rows = []

    for episode in range(1, N_EPISODES + 1):
        obs = env.reset()

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, dones, infos = env.step(action)

            done = bool(dones[0])
            info = infos[0]

            action_value = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)

            fluid_bin, vasopressor_bin = decode_action(action_value)

            total_reward += reward_value
            steps += 1

            transition_rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "episode": episode,
                "step": steps,
                "action": action_value,
                "fluid_bin": fluid_bin,
                "vasopressor_bin": vasopressor_bin,
                "reward": reward_value,
                "done": int(done),
                "clinician_action": int(info.get("clinician_action", -1)),
                "clinician_fluid_bin": int(info.get("clinician_fluid_bin", -1)),
                "clinician_vasopressor_bin": int(info.get("clinician_vasopressor_bin", -1)),
                "mortality": int(info.get("mortality", -1)),
                "lactate": float(info.get("lactate", np.nan)),
                "next_lactate": float(info.get("next_lactate", np.nan)),
                "map": float(info.get("map", np.nan)),
                "next_map": float(info.get("next_map", np.nan)),
                "sofa_proxy": float(info.get("sofa_proxy", np.nan)),
                "next_sofa_proxy": float(info.get("next_sofa_proxy", np.nan)),
            })

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(
            f"{algorithm} Seed {seed} | Episode {episode}/{N_EPISODES}: "
            f"Reward = {total_reward:.3f}, Length = {steps}"
        )

    env.close()

    episode_rewards = np.asarray(episode_rewards, dtype=float)
    episode_lengths = np.asarray(episode_lengths, dtype=float)

    transition_df = pd.DataFrame(transition_rows)

    action_counts = transition_df["action"].value_counts().sort_index()
    action_dist_rows = []

    for action in range(25):
        count = int(action_counts.get(action, 0))
        fluid_bin, vasopressor_bin = decode_action(action)

        action_dist_rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "action": action,
            "fluid_bin": fluid_bin,
            "vasopressor_bin": vasopressor_bin,
            "count": count,
            "percent": float(count / max(len(transition_df), 1)),
        })

    action_dist_df = pd.DataFrame(action_dist_rows)

    severity_df = create_severity_subgroup_summary(
        transition_df=transition_df,
        algorithm=algorithm,
        seed=seed
    )

    metrics = {
        "algorithm": algorithm,
        "phase": PHASE,
        "seed": seed,
        "n_episodes": N_EPISODES,
        "action_space": 25,
        "model_path": model_path + ".zip",
        "vecnormalize_path": vec_path,
        "data_path": DATA_PATH,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "reward_q25": float(np.percentile(episode_rewards, 25)),
        "median_reward": float(np.median(episode_rewards)),
        "reward_q75": float(np.percentile(episode_rewards, 75)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "mean_fluid_bin": float(transition_df["fluid_bin"].mean()),
        "mean_vasopressor_bin": float(transition_df["vasopressor_bin"].mean()),
        "mean_clinician_fluid_bin": float(transition_df["clinician_fluid_bin"].replace(-1, np.nan).mean()),
        "mean_clinician_vasopressor_bin": float(transition_df["clinician_vasopressor_bin"].replace(-1, np.nan).mean()),
        "action_match_rate": float(
            np.mean(
                transition_df["action"].values
                == transition_df["clinician_action"].values
            )
        ),
        "timestamp": timestamp,
    }

    return metrics, transition_df, action_dist_df, severity_df, episode_rewards


def create_severity_subgroup_summary(transition_df: pd.DataFrame, algorithm: str, seed: int):
    df = transition_df.copy()

    if "sofa_proxy" not in df.columns:
        return pd.DataFrame()

    df = df[np.isfinite(df["sofa_proxy"])].copy()

    if df.empty:
        return pd.DataFrame()

    try:
        df["severity_group"] = pd.qcut(
            df["sofa_proxy"],
            q=3,
            labels=["Low", "Moderate", "High"],
            duplicates="drop"
        )
    except ValueError:
        return pd.DataFrame()

    rows = []

    for group in ["Low", "Moderate", "High"]:
        g = df[df["severity_group"].astype(str) == group]

        if g.empty:
            continue

        rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "severity_group": group,
            "n_transitions": int(len(g)),
            "mean_sofa_proxy": float(g["sofa_proxy"].mean()),
            "mean_reward": float(g["reward"].mean()),
            "mean_fluid_bin": float(g["fluid_bin"].mean()),
            "mean_vasopressor_bin": float(g["vasopressor_bin"].mean()),
            "mean_clinician_fluid_bin": float(g["clinician_fluid_bin"].replace(-1, np.nan).mean()),
            "mean_clinician_vasopressor_bin": float(g["clinician_vasopressor_bin"].replace(-1, np.nan).mean()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_episode_rewards(all_rewards_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    for (algorithm, seed), group in all_rewards_df.groupby(["algorithm", "seed"]):
        plt.plot(
            group["episode"],
            group["episode_reward"],
            marker="o",
            linestyle="-",
            label=f"{algorithm} Seed {seed}"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.title("PPO and A2C Temporal 25-Action Policy Evaluation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"{FIGURES_DIR}/ppo_a2c_episode_rewards_{timestamp}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved episode reward plot to {plot_path}")


def plot_action_distribution(all_action_df: pd.DataFrame):
    for algorithm in all_action_df["algorithm"].unique():
        df_alg = all_action_df[all_action_df["algorithm"] == algorithm]

        plt.figure(figsize=(10, 5))

        for seed, group in df_alg.groupby("seed"):
            plt.plot(
                group["action"],
                group["percent"],
                marker="o",
                label=f"Seed {seed}"
            )

        plt.xlabel("Action index")
        plt.ylabel("Action proportion")
        plt.title(f"{algorithm} Selected Action Distribution")
        plt.xticks(range(25), rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = f"{FIGURES_DIR}/{algorithm.lower()}_action_distribution_{timestamp}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Saved {algorithm} action distribution plot to {plot_path}")


def plot_severity_vasopressor(all_severity_df: pd.DataFrame):
    if all_severity_df.empty:
        return

    summary = (
        all_severity_df
        .groupby(["algorithm", "severity_group"], as_index=False)
        ["mean_vasopressor_bin"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    severity_order = ["Low", "Moderate", "High"]

    for algorithm in summary["algorithm"].unique():
        alg_df = summary[summary["algorithm"] == algorithm].copy()
        alg_df["severity_group"] = pd.Categorical(
            alg_df["severity_group"],
            categories=severity_order,
            ordered=True
        )
        alg_df = alg_df.sort_values("severity_group")

        plt.plot(
            alg_df["severity_group"].astype(str),
            alg_df["mean_vasopressor_bin"],
            marker="o",
            label=algorithm
        )

    plt.xlabel("Severity subgroup")
    plt.ylabel("Mean selected vasopressor bin")
    plt.title("PPO and A2C Vasopressor Intensity Across Severity Subgroups")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"{FIGURES_DIR}/ppo_a2c_severity_vasopressor_{timestamp}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved severity vasopressor plot to {plot_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Trajectory data not found at {DATA_PATH}. "
            "Update DATA_PATH to match your project structure."
        )

    all_metrics = []
    all_episode_rows = []
    all_transition_dfs = []
    all_action_dfs = []
    all_severity_dfs = []

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            metrics, transition_df, action_df, severity_df, episode_rewards = evaluate_single_model(
                algorithm=algorithm,
                seed=seed,
                device=device
            )

            all_metrics.append(metrics)
            all_transition_dfs.append(transition_df)
            all_action_dfs.append(action_df)

            if severity_df is not None and not severity_df.empty:
                all_severity_dfs.append(severity_df)

            for i, reward in enumerate(episode_rewards, start=1):
                all_episode_rows.append({
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": i,
                    "episode_reward": float(reward),
                    "episode_length": int(
                        transition_df[transition_df["episode"] == i]["step"].max()
                    ),
                })

    metrics_df = pd.DataFrame(all_metrics)
    episode_df = pd.DataFrame(all_episode_rows)
    transition_all_df = pd.concat(all_transition_dfs, ignore_index=True)
    action_all_df = pd.concat(all_action_dfs, ignore_index=True)

    if all_severity_dfs:
        severity_all_df = pd.concat(all_severity_dfs, ignore_index=True)
    else:
        severity_all_df = pd.DataFrame()

    metrics_path = f"{RESULTS_DIR}/ppo_a2c_evaluation_metrics_{timestamp}.csv"
    episode_path = f"{RESULTS_DIR}/ppo_a2c_episode_rewards_{timestamp}.csv"
    transition_path = f"{RESULTS_DIR}/ppo_a2c_transition_predictions_{timestamp}.csv"
    action_path = f"{RESULTS_DIR}/ppo_a2c_action_distribution_{timestamp}.csv"
    severity_path = f"{RESULTS_DIR}/ppo_a2c_severity_subgroup_policy_{timestamp}.csv"
    json_path = f"{RESULTS_DIR}/ppo_a2c_evaluation_metrics_{timestamp}.json"

    metrics_df.to_csv(metrics_path, index=False)
    episode_df.to_csv(episode_path, index=False)
    transition_all_df.to_csv(transition_path, index=False)
    action_all_df.to_csv(action_path, index=False)

    if not severity_all_df.empty:
        severity_all_df.to_csv(severity_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    plot_episode_rewards(episode_df)
    plot_action_distribution(action_all_df)
    plot_severity_vasopressor(severity_all_df)

    print("\n[DONE] PPO/A2C evaluation completed successfully.")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved episode rewards: {episode_path}")
    print(f"Saved transition predictions: {transition_path}")
    print(f"Saved action distribution: {action_path}")
    if not severity_all_df.empty:
        print(f"Saved severity subgroup policy: {severity_path}")
    print(f"Saved metrics JSON: {json_path}")


if __name__ == "__main__":
    main()