#!/usr/bin/env python3
"""
train_ppo_ablation.py

Train PPO ablation variants for the sepsis temporal RL project.

Run from scripts/new.

Quick smoke test:
    python .\train_ppo_ablation.py --variants no_clinician_penalty --seeds 1 --timesteps 10000

Train one variant across five seeds:
    python .\train_ppo_ablation.py --variants no_clinician_penalty --seeds 1 2 3 4 5 --timesteps 5000000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from sepsis_ablation_env import SepsisAblationEnv, VALID_ABLATIONS


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent

DEFAULT_DATA_PATH = SCRIPTS_DIR / "data" / "sepsis_trajectories.csv"
DEFAULT_MODEL_DIR = SCRIPTS_DIR / "models_ablation"
DEFAULT_RESULTS_DIR = SCRIPTS_DIR / "results_ablation"
DEFAULT_LOG_DIR = SCRIPTS_DIR / "logs_ablation"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TIMESTEPS = 5_000_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(data_path: Path, ablation_name: str, seed: int):
    def _init():
        env = SepsisAblationEnv(data_path=str(data_path), ablation_name=ablation_name)
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(data_path: Path, ablation_name: str, seed: int, norm_reward: bool):
    env = DummyVecEnv([make_env(data_path, ablation_name, seed)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0, gamma=0.99)
    return env


def train_one_variant_seed(data_path, model_dir, results_dir, log_dir, ablation_name, seed, timesteps, device):
    set_seed(seed)
    norm_reward = ablation_name != "no_reward_normalization"

    variant_model_dir = model_dir / ablation_name
    variant_results_dir = results_dir / ablation_name
    variant_log_dir = log_dir / ablation_name
    variant_model_dir.mkdir(parents=True, exist_ok=True)
    variant_results_dir.mkdir(parents=True, exist_ok=True)
    variant_log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[INFO] Training PPO ablation variant: {ablation_name} | Seed: {seed}")
    print(f"[INFO] Timesteps: {timesteps}")
    print(f"[INFO] Reward normalization: {norm_reward}")
    print("=" * 80)

    env = build_vec_env(data_path, ablation_name, seed, norm_reward)

    checkpoint_dir = variant_model_dir / f"ppo_{ablation_name}_checkpoints_seed_{seed}"
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_{ablation_name}_seed_{seed}",
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
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
        tensorboard_log=str(variant_log_dir / f"PPO_{ablation_name}_seed_{seed}"),
        device=device,
    )

    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)

    model_path = variant_model_dir / f"ppo_{ablation_name}_seed_{seed}"
    vec_path = variant_model_dir / f"ppo_{ablation_name}_seed_{seed}_vecnormalize.pkl"
    model.save(str(model_path))
    env.save(str(vec_path))

    summary = {
        "algorithm": "PPO",
        "experiment": "ablation",
        "ablation_name": ablation_name,
        "seed": seed,
        "timesteps": timesteps,
        "action_space": 25,
        "environment": "SepsisAblationEnv",
        "data_path": str(data_path),
        "model_path": str(model_path) + ".zip",
        "vecnormalize_path": str(vec_path),
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "network_architecture": [256, 256],
        "norm_obs": True,
        "norm_reward": norm_reward,
        "clip_obs": 10.0,
        "device": device,
    }

    summary_path = variant_results_dir / f"ppo_{ablation_name}_seed_{seed}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    env.close()

    print(f"[DONE] Saved model: {model_path}.zip")
    print(f"[DONE] Saved VecNormalize: {vec_path}")
    print(f"[DONE] Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO ablation variants.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--log_dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--variants", nargs="+", default=["no_clinician_penalty"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    log_dir = Path(args.log_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found: {data_path}")

    for variant in args.variants:
        if variant not in VALID_ABLATIONS:
            raise ValueError(f"Invalid variant: {variant}. Valid variants: {sorted(VALID_ABLATIONS)}")

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    for variant in args.variants:
        for seed in args.seeds:
            train_one_variant_seed(
                data_path=data_path,
                model_dir=model_dir,
                results_dir=results_dir,
                log_dir=log_dir,
                ablation_name=variant,
                seed=seed,
                timesteps=args.timesteps,
                device=device,
            )

    print("\n[DONE] PPO ablation training completed.")


if __name__ == "__main__":
    main()
