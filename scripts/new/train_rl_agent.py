#!/usr/bin/env python3
"""
train_ppo_temporal_25.py

Train PPO temporal reinforcement learning agents for sequential sepsis
treatment optimization using the same SepsisTrajectoryEnv used for A2C
and the env-aligned CQL offline dataset.

Environment design:
    One episode = one ICU stay
    One timestep = one 4-hour clinical window
    Action space = 25 fluid-vasopressor treatment actions

This script is aligned with:
    - sepsis_temporal_env.py
    - prepare_offline_dataset_env_aligned.py
    - train_cql_agent.py
    - evaluate_cql_agent.py
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from sepsis_temporal_env import SepsisTrajectoryEnv

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TIMESTEPS = 5_000_000
SEEDS = [1,2,3,4,5]

DATA_PATH = "../data/sepsis_trajectories.csv"

MODEL_DIR = "../models"
RESULTS_DIR = "../results"
LOG_DIR = "../logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------

def make_env(seed: int):
    def _init():
        env = SepsisTrajectoryEnv(data_path=DATA_PATH)
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(seed: int):
    env = DummyVecEnv([make_env(seed)])
    env = VecMonitor(env)

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )

    return env


# ---------------------------------------------------------------------
# Main PPO training loop
# ---------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Trajectory data not found at {DATA_PATH}. "
            "Update DATA_PATH to match your project structure."
        )

    for seed in SEEDS:
        print(f"\nTraining PPO temporal 25-action model with SEED = {seed}\n")

        set_seed(seed)

        env = build_vec_env(seed)

        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=f"{MODEL_DIR}/ppo_temporal_25_checkpoints_seed_{seed}",
            name_prefix=f"ppo_temporal_25_seed_{seed}",
        )

        policy_kwargs = dict(
            net_arch=[256, 256]
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            seed=seed,
            tensorboard_log=f"{LOG_DIR}/PPO_temporal_25_seed_{seed}",
            device=device,
        )

        model.learn(
            total_timesteps=TIMESTEPS,
            callback=checkpoint_callback,
        )

        model_path = f"{MODEL_DIR}/ppo_temporal_25_seed_{seed}"
        vec_path = f"{MODEL_DIR}/ppo_temporal_25_seed_{seed}_vecnormalize.pkl"

        model.save(model_path)
        env.save(vec_path)

        summary = {
            "algorithm": "PPO",
            "phase": "temporal_25_action",
            "seed": seed,
            "timesteps": TIMESTEPS,
            "action_space": 25,
            "environment": "SepsisTrajectoryEnv",
            "data_path": DATA_PATH,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "network_architecture": [256, 256],
            "vecnormalize": True,
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": 10.0,
            "device": device,
        }

        summary_path = f"{RESULTS_DIR}/ppo_temporal_25_seed_{seed}_summary.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        env.close()

        print(f"Saved PPO model: {model_path}.zip")
        print(f"Saved PPO VecNormalize: {vec_path}")
        print(f"Saved PPO summary: {summary_path}")

    print("\n[DONE] PPO training completed for all seeds.")


if __name__ == "__main__":
    main()