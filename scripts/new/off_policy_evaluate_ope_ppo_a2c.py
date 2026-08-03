#!/usr/bin/env python3
"""
evaluate_ope_ppo_a2c.py

Off-policy-style evaluation diagnostics for trained PPO and A2C temporal
25-action policies.

This script:
    - Loads trained PPO and A2C models
    - Loads matching VecNormalize statistics
    - Uses the same SepsisTrajectoryEnv
    - Runs stochastic policy rollouts
    - Computes IS, WIS, and DM-style estimates
    - Evaluates Seeds 3, 4, and 5
    - Saves OPE metrics without overwriting older files

Important note:
    This is an OPE-style diagnostic using a uniform behavior-policy assumption
    over the 25-action space. It should be reported carefully as an approximate
    retrospective policy-evaluation diagnostic, not as prospective clinical
    validation.

Aligned with:
    - sepsis_temporal_env.py
    - train_ppo_temporal_25.py
    - train_a2c_temporal_25.py
    - evaluate_ppo_a2c_agents.py
    - train_cql_agent.py
    - evaluate_cql_agent.py
"""

import os
import time
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sepsis_temporal_env import SepsisTrajectoryEnv

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ALGORITHMS = ["PPO", "A2C"]
SEEDS = [1, 2, 3, 4, 5 ]

PHASE = "temporal_25_action"

DATA_PATH = "../data/sepsis_trajectories.csv"
MODEL_DIR = "../models"
RESULTS_DIR = "../results"

N_EVAL_EPISODES = 100
GAMMA = 0.99

NUM_ACTIONS = 25
BEHAVIOR_PROB = 1.0 / NUM_ACTIONS

os.makedirs(RESULTS_DIR, exist_ok=True)

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
# Environment and model loading
# ---------------------------------------------------------------------

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
# OPE helper functions
# ---------------------------------------------------------------------

def get_action_probability(model, obs, action_int: int, device: str) -> float:
    """
    Get model probability for the selected action.
    Works for PPO and A2C discrete policies.
    """
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)

    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]

    action_prob = float(probs[action_int])
    action_prob = max(action_prob, 1e-12)

    return action_prob


def discounted_return(rewards, gamma: float) -> float:
    rewards = np.asarray(rewards, dtype=float)
    discounts = np.power(gamma, np.arange(len(rewards)))
    return float(np.sum(discounts * rewards))


def safe_is_estimate(log_weights: np.ndarray, returns: np.ndarray) -> float:
    """
    Ordinary IS estimate using clipped exponentiation to avoid hard overflow.
    IS may still become extremely large, which is expected in long-horizon RL.
    """
    clipped_log_weights = np.clip(log_weights, -700, 700)
    weights = np.exp(clipped_log_weights)
    return float(np.mean(weights * returns))


def safe_wis_estimate(log_weights: np.ndarray, returns: np.ndarray) -> float:
    """
    Weighted IS estimate computed in a numerically stable manner.
    """
    max_log_weight = np.max(log_weights)
    stabilized_weights = np.exp(log_weights - max_log_weight)

    denominator = np.sum(stabilized_weights)

    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")

    return float(np.sum(stabilized_weights * returns) / denominator)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_single_algorithm_seed(algorithm: str, seed: int, device: str):
    print(f"\nStarting {algorithm} temporal 25-action OPE evaluation | Seed {seed}\n")

    set_seed(seed)

    model, env, model_path, vec_path = load_model_and_env(
        algorithm=algorithm,
        seed=seed,
        device=device
    )

    episode_rows = []

    log_importance_weights = []
    raw_importance_weights = []
    returns = []
    dm_proxy_returns = []
    episode_lengths = []

    for ep in range(1, N_EVAL_EPISODES + 1):
        obs = env.reset()
        done = False

        rewards = []
        action_probs = []
        selected_actions = []

        steps = 0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=False
            )

            action_int = int(action[0]) if isinstance(action, np.ndarray) else int(action)

            action_prob = get_action_probability(
                model=model,
                obs=obs,
                action_int=action_int,
                device=device
            )

            obs, reward, dones, infos = env.step(np.array([action_int]))

            done = bool(dones[0])
            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)

            rewards.append(reward_value)
            action_probs.append(action_prob)
            selected_actions.append(action_int)

            steps += 1

        ep_return = discounted_return(rewards, gamma=GAMMA)

        # Log-space importance ratio:
        # product_t pi(a_t|s_t) / b(a_t|s_t)
        # where b is approximated as uniform over 25 actions.
        log_weight = float(
            np.sum(np.log(np.asarray(action_probs) + 1e-12) - np.log(BEHAVIOR_PROB))
        )

        clipped_raw_weight = float(np.exp(np.clip(log_weight, -700, 700)))

        dm_proxy = float(np.mean(rewards) * len(rewards)) if len(rewards) > 0 else 0.0

        log_importance_weights.append(log_weight)
        raw_importance_weights.append(clipped_raw_weight)
        returns.append(ep_return)
        dm_proxy_returns.append(dm_proxy)
        episode_lengths.append(steps)

        episode_rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "episode": ep,
            "discounted_return": ep_return,
            "dm_proxy_return": dm_proxy,
            "episode_length": steps,
            "log_importance_weight": log_weight,
            "importance_weight_clipped": clipped_raw_weight,
            "mean_action_probability": float(np.mean(action_probs)),
            "min_action_probability": float(np.min(action_probs)),
            "max_action_probability": float(np.max(action_probs)),
            "mean_selected_action": float(np.mean(selected_actions)),
            "mean_selected_fluid_bin": float(np.mean(np.asarray(selected_actions) // 5)),
            "mean_selected_vasopressor_bin": float(np.mean(np.asarray(selected_actions) % 5)),
        })

        print(
            f"{algorithm} Seed {seed} | Episode {ep}/{N_EVAL_EPISODES}: "
            f"Return={ep_return:.3f}, Length={steps}, LogW={log_weight:.2f}"
        )

    env.close()

    log_importance_weights = np.asarray(log_importance_weights, dtype=float)
    raw_importance_weights = np.asarray(raw_importance_weights, dtype=float)
    returns = np.asarray(returns, dtype=float)
    dm_proxy_returns = np.asarray(dm_proxy_returns, dtype=float)
    episode_lengths = np.asarray(episode_lengths, dtype=float)

    true_return = float(np.mean(returns))
    is_estimate = safe_is_estimate(log_importance_weights, returns)
    wis_estimate = safe_wis_estimate(log_importance_weights, returns)
    dm_estimate = float(np.mean(dm_proxy_returns))

    metrics = {
        "algorithm": algorithm,
        "phase": PHASE,
        "seed": seed,
        "n_eval_episodes": N_EVAL_EPISODES,
        "gamma": GAMMA,
        "action_space": NUM_ACTIONS,
        "behavior_policy_assumption": "uniform_random_over_25_actions",
        "behavior_prob": BEHAVIOR_PROB,
        "model_path": model_path + ".zip",
        "vecnormalize_path": vec_path,
        "data_path": DATA_PATH,
        "true_average_return_from_policy_rollout": true_return,
        "is_estimate": is_estimate,
        "wis_estimate": wis_estimate,
        "dm_proxy_estimate": dm_estimate,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_q25": float(np.percentile(returns, 25)),
        "return_median": float(np.median(returns)),
        "return_q75": float(np.percentile(returns, 75)),
        "return_max": float(np.max(returns)),
        "mean_log_importance_weight": float(np.mean(log_importance_weights)),
        "std_log_importance_weight": float(np.std(log_importance_weights)),
        "max_log_importance_weight": float(np.max(log_importance_weights)),
        "mean_importance_weight_clipped": float(np.mean(raw_importance_weights)),
        "max_importance_weight_clipped": float(np.max(raw_importance_weights)),
        "timestamp": timestamp,
    }

    episode_df = pd.DataFrame(episode_rows)

    print(f"\n=== {algorithm} Temporal 25-Action OPE Results | Seed {seed} ===")
    print(f"True average rollout return: {true_return:.4f}")
    print(f"IS estimate: {is_estimate:.4e}")
    print(f"WIS estimate: {wis_estimate:.4f}")
    print(f"DM proxy estimate: {dm_estimate:.4f}")
    print(f"Mean episode length: {metrics['mean_episode_length']:.2f}")

    return metrics, episode_df


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
    all_episode_dfs = []

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            metrics, episode_df = evaluate_single_algorithm_seed(
                algorithm=algorithm,
                seed=seed,
                device=device
            )

            all_metrics.append(metrics)
            all_episode_dfs.append(episode_df)

    metrics_df = pd.DataFrame(all_metrics)
    episode_df_all = pd.concat(all_episode_dfs, ignore_index=True)

    metrics_csv_path = f"{RESULTS_DIR}/ppo_a2c_ope_metrics_{timestamp}.csv"
    metrics_json_path = f"{RESULTS_DIR}/ppo_a2c_ope_metrics_{timestamp}.json"
    episode_csv_path = f"{RESULTS_DIR}/ppo_a2c_ope_episode_details_{timestamp}.csv"

    metrics_df.to_csv(metrics_csv_path, index=False)
    episode_df_all.to_csv(episode_csv_path, index=False)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print("\n[DONE] PPO/A2C OPE evaluation completed successfully.")
    print(f"Saved OPE metrics CSV: {metrics_csv_path}")
    print(f"Saved OPE metrics JSON: {metrics_json_path}")
    print(f"Saved OPE episode details: {episode_csv_path}")


if __name__ == "__main__":
    main()