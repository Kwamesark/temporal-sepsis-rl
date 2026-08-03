#!/usr/bin/env python3
"""
evaluate_ppo_ablation.py

Evaluate trained PPO ablation variants.

Run from scripts/new.

Example:
    python .\evaluate_ppo_ablation.py --variants no_clinician_penalty --seeds 1 2 3 4 5 --n_episodes 200
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sepsis_ablation_env import SepsisAblationEnv, VALID_ABLATIONS


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent

DEFAULT_DATA_PATH = SCRIPTS_DIR / "data" / "sepsis_trajectories.csv"
DEFAULT_MODEL_DIR = SCRIPTS_DIR / "models_ablation"
DEFAULT_RESULTS_DIR = SCRIPTS_DIR / "results_ablation"
DEFAULT_FIGURES_DIR = SCRIPTS_DIR / "figures_ablation"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_VARIANTS = [
    "full",
    "no_clinician_penalty",
    "no_severity_treatment",
    "no_sofa_proxy",
    "no_reward_normalization",
]
DEFAULT_N_EPISODES = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_action(action: int):
    action = int(action)
    return action // 5, action % 5


def build_raw_env(data_path: Path, ablation_name: str):
    return DummyVecEnv([lambda: SepsisAblationEnv(data_path=str(data_path), ablation_name=ablation_name)])


def load_model_and_env(data_path: Path, model_dir: Path, ablation_name: str, seed: int, device: str):
    variant_model_dir = model_dir / ablation_name
    model_path = variant_model_dir / f"ppo_{ablation_name}_seed_{seed}"
    vec_path = variant_model_dir / f"ppo_{ablation_name}_seed_{seed}_vecnormalize.pkl"

    if not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}.zip")
    if not vec_path.exists():
        raise FileNotFoundError(f"VecNormalize not found: {vec_path}")

    raw_env = build_raw_env(data_path, ablation_name)
    env = VecNormalize.load(str(vec_path), raw_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_path), env=env, device=device)
    return model, env, str(model_path), str(vec_path)


def evaluate_variant_seed(data_path, model_dir, ablation_name, seed, n_episodes, device):
    print(f"\n[INFO] Evaluating ablation={ablation_name} | seed={seed}")
    set_seed(seed)

    model, env, model_path, vec_path = load_model_and_env(data_path, model_dir, ablation_name, seed, device)

    episode_rewards = []
    episode_lengths = []
    transition_rows = []

    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            info = infos[0]

            action_int = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            fluid_bin, vaso_bin = decode_action(action_int)

            total_reward += reward_value
            steps += 1

            clinician_action = int(info.get("clinician_action", -1))
            clinician_fluid_bin = int(info.get("clinician_fluid_bin", -1))
            clinician_vaso_bin = int(info.get("clinician_vasopressor_bin", -1))

            transition_rows.append({
                "ablation_name": ablation_name,
                "seed": seed,
                "episode": episode,
                "step": steps,
                "action": action_int,
                "fluid_bin": fluid_bin,
                "vasopressor_bin": vaso_bin,
                "reward": reward_value,
                "clinician_action": clinician_action,
                "clinician_fluid_bin": clinician_fluid_bin,
                "clinician_vasopressor_bin": clinician_vaso_bin,
                "action_match": int(action_int == clinician_action),
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

    env.close()

    episode_rewards = np.asarray(episode_rewards, dtype=float)
    episode_lengths = np.asarray(episode_lengths, dtype=float)
    transition_df = pd.DataFrame(transition_rows)

    metrics = {
        "algorithm": "PPO",
        "experiment": "ablation",
        "ablation_name": ablation_name,
        "seed": seed,
        "n_episodes": n_episodes,
        "model_path": model_path + ".zip",
        "vecnormalize_path": vec_path,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "mean_fluid_bin": float(transition_df["fluid_bin"].mean()),
        "mean_vasopressor_bin": float(transition_df["vasopressor_bin"].mean()),
        "mean_clinician_fluid_bin": float(transition_df["clinician_fluid_bin"].replace(-1, np.nan).mean()),
        "mean_clinician_vasopressor_bin": float(transition_df["clinician_vasopressor_bin"].replace(-1, np.nan).mean()),
        "action_match_rate": float(transition_df["action_match"].mean()),
    }

    return metrics, transition_df, episode_rewards


def plot_ablation_rewards(summary_df: pd.DataFrame, figures_dir: Path):
    if summary_df.empty:
        return
    grouped = summary_df.groupby("ablation_name", as_index=False).agg(
        mean_reward=("mean_reward", "mean"),
        std_reward=("mean_reward", "std"),
    )
    plt.figure(figsize=(10, 5))
    plt.bar(grouped["ablation_name"], grouped["mean_reward"], yerr=grouped["std_reward"])
    plt.xlabel("Ablation variant")
    plt.ylabel("Mean deterministic rollout reward")
    plt.title("PPO Ablation Study: Mean Reward Across Seeds")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = figures_dir / "ppo_ablation_reward_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved reward plot: {path}")


def plot_ablation_treatment_bins(summary_df: pd.DataFrame, figures_dir: Path):
    if summary_df.empty:
        return
    grouped = summary_df.groupby("ablation_name", as_index=False).agg(
        mean_fluid_bin=("mean_fluid_bin", "mean"),
        mean_vasopressor_bin=("mean_vasopressor_bin", "mean"),
    )
    x = np.arange(len(grouped))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, grouped["mean_fluid_bin"], width=width, label="Fluid")
    plt.bar(x + width / 2, grouped["mean_vasopressor_bin"], width=width, label="Vasopressor")
    plt.xlabel("Ablation variant")
    plt.ylabel("Mean selected intensity bin")
    plt.title("PPO Ablation Study: Treatment Intensity")
    plt.xticks(x, grouped["ablation_name"], rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = figures_dir / "ppo_ablation_treatment_bins.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved treatment-bin plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO ablation variants.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--figures_dir", type=str, default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n_episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found: {data_path}")
    for variant in args.variants:
        if variant not in VALID_ABLATIONS:
            raise ValueError(f"Invalid variant: {variant}. Valid variants: {sorted(VALID_ABLATIONS)}")

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    all_metrics = []
    all_transition_dfs = []
    episode_rows = []

    for variant in args.variants:
        for seed in args.seeds:
            try:
                metrics, transition_df, episode_rewards = evaluate_variant_seed(
                    data_path, model_dir, variant, seed, args.n_episodes, device
                )
                all_metrics.append(metrics)
                all_transition_dfs.append(transition_df)
                for idx, reward in enumerate(episode_rewards, start=1):
                    episode_rows.append({
                        "ablation_name": variant,
                        "seed": seed,
                        "episode": idx,
                        "episode_reward": float(reward),
                    })
            except FileNotFoundError as exc:
                print(f"[WARN] Skipping {variant} seed {seed}: {exc}")

    if not all_metrics:
        raise RuntimeError("No ablation models were evaluated. Check model paths.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_df = pd.DataFrame(all_metrics)
    transition_df_all = pd.concat(all_transition_dfs, ignore_index=True)
    episode_df = pd.DataFrame(episode_rows)

    metrics_path = results_dir / f"ppo_ablation_evaluation_metrics_{timestamp}.csv"
    transition_path = results_dir / f"ppo_ablation_transition_predictions_{timestamp}.csv"
    episode_path = results_dir / f"ppo_ablation_episode_rewards_{timestamp}.csv"

    metrics_df.to_csv(metrics_path, index=False)
    transition_df_all.to_csv(transition_path, index=False)
    episode_df.to_csv(episode_path, index=False)

    aggregate = metrics_df.groupby("ablation_name", as_index=False).agg(
        mean_reward=("mean_reward", "mean"),
        std_reward=("mean_reward", "std"),
        mean_episode_length=("mean_episode_length", "mean"),
        mean_fluid_bin=("mean_fluid_bin", "mean"),
        mean_vasopressor_bin=("mean_vasopressor_bin", "mean"),
        action_match_rate=("action_match_rate", "mean"),
    )
    aggregate_path = results_dir / "ppo_ablation_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)

    with open(results_dir / f"ppo_ablation_evaluation_metrics_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    plot_ablation_rewards(metrics_df, figures_dir)
    plot_ablation_treatment_bins(metrics_df, figures_dir)

    print("\n=== PPO Ablation Aggregate Summary ===")
    print(aggregate.to_string(index=False))
    print("\n[DONE] Ablation evaluation completed.")
    print(f"Metrics: {metrics_path}")
    print(f"Aggregate summary: {aggregate_path}")
    print(f"Transitions: {transition_path}")
    print(f"Episodes: {episode_path}")


if __name__ == "__main__":
    main()
