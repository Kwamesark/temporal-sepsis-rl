#!/usr/bin/env python3
"""
analyze_actions.py

Analyze treatment-action behavior for trained PPO, A2C, and CQL temporal
25-action policies after 5-seed training.

This version fixes:
    - 5-seed setup: seeds 1, 2, 3, 4, 5
    - local import when running from scripts/new
    - CQL integration across all cql_evaluation_results_seed* folders
    - duplicate heatmap-cell error by using grouped aggregation + pivot_table
    - avoids loading duplicate CQL files from cql_results_seed* training folders

Run from scripts/new:
    python .\\analyze_actions.py

If you used 5-seed folders:
    python .\\analyze_actions.py --model_dir ..\\models_5seed --results_dir ..\\results_5seed --figures_dir ..\\figures_5seed

If your PPO/A2C models are in the original folders:
    python .\\analyze_actions.py --model_dir ..\\models --results_dir ..\\results --figures_dir ..\\figures

For quick testing without CQL:
    python .\\analyze_actions.py --model_dir ..\\models_5seed --results_dir ..\\results_5seed --figures_dir ..\\figures_5seed --skip_cql
"""

import argparse
import glob
import random
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sepsis_temporal_env import SepsisTrajectoryEnv
except ModuleNotFoundError:
    from scripts.new.sepsis_temporal_env import SepsisTrajectoryEnv


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent

DEFAULT_DATA_PATH = SCRIPTS_DIR / "data" / "sepsis_trajectories.csv"

DEFAULT_MODEL_DIR = SCRIPTS_DIR / (
    "models_5seed" if (SCRIPTS_DIR / "models_5seed").exists() else "models"
)
DEFAULT_RESULTS_DIR = SCRIPTS_DIR / (
    "results_5seed" if (SCRIPTS_DIR / "results_5seed").exists() else "results"
)
DEFAULT_FIGURES_DIR = SCRIPTS_DIR / (
    "figures_5seed" if (SCRIPTS_DIR / "figures_5seed").exists() else "figures"
)

SEEDS = [1, 2, 3, 4, 5]
ALGORITHMS = ["PPO", "A2C"]
N_EPISODES = 200


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

def decode_action(action: int) -> Tuple[int, int]:
    action = int(action)
    return action // 5, action % 5


def extract_seed_from_path(path: Path) -> int:
    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1


def build_raw_env(data_path: Path):
    return DummyVecEnv([lambda: SepsisTrajectoryEnv(data_path=str(data_path))])


def load_model_and_env(
    algorithm: str,
    seed: int,
    data_path: Path,
    model_dir: Path,
    device: str,
):
    algorithm_lower = algorithm.lower()

    model_path = model_dir / f"{algorithm_lower}_temporal_25_seed_{seed}"
    vec_path = model_dir / f"{algorithm_lower}_temporal_25_seed_{seed}_vecnormalize.pkl"

    if not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(f"Model file not found: {model_path}.zip")

    if not vec_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vec_path}")

    raw_env = build_raw_env(data_path)
    env = VecNormalize.load(str(vec_path), raw_env)

    env.training = False
    env.norm_reward = False

    if algorithm.upper() == "PPO":
        model = PPO.load(str(model_path), env=env, device=device)
    elif algorithm.upper() == "A2C":
        model = A2C.load(str(model_path), env=env, device=device)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return model, env, str(model_path), str(vec_path)


# ---------------------------------------------------------------------
# PPO/A2C action collection
# ---------------------------------------------------------------------

def collect_policy_actions(
    algorithm: str,
    seed: int,
    data_path: Path,
    model_dir: Path,
    device: str,
    n_episodes: int,
) -> pd.DataFrame:
    print(f"\n[INFO] Analyzing {algorithm} policy actions for seed {seed}...")

    set_seed(seed)

    model, env, model_path, vec_path = load_model_and_env(
        algorithm=algorithm,
        seed=seed,
        data_path=data_path,
        model_dir=model_dir,
        device=device,
    )

    rows = []

    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        timestep = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            action_int = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            fluid_bin, vasopressor_bin = decode_action(action_int)

            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])

            info = infos[0]
            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)

            clinician_action = int(info.get("clinician_action", -1))
            clinician_fluid_bin = int(info.get("clinician_fluid_bin", -1))
            clinician_vasopressor_bin = int(info.get("clinician_vasopressor_bin", -1))

            rows.append({
                "algorithm": algorithm.upper(),
                "seed": int(seed),
                "episode": int(episode),
                "timestep": int(timestep),
                "action": int(action_int),
                "fluid_bin": int(fluid_bin),
                "vasopressor_bin": int(vasopressor_bin),
                "reward": float(reward_value),
                "clinician_action": clinician_action,
                "clinician_fluid_bin": clinician_fluid_bin,
                "clinician_vasopressor_bin": clinician_vasopressor_bin,
                "action_match": int(action_int == clinician_action),
                "mortality": int(info.get("mortality", -1)),
                "lactate": float(info.get("lactate", np.nan)),
                "next_lactate": float(info.get("next_lactate", np.nan)),
                "map": float(info.get("map", np.nan)),
                "next_map": float(info.get("next_map", np.nan)),
                "sofa_proxy": float(info.get("sofa_proxy", np.nan)),
                "next_sofa_proxy": float(info.get("next_sofa_proxy", np.nan)),
                "model_path": model_path,
                "vecnormalize_path": vec_path,
            })

            timestep += 1

    env.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------

def make_action_summary(action_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (algorithm, seed), group in action_df.groupby(["algorithm", "seed"]):
        total = max(len(group), 1)
        counts = group["action"].value_counts().sort_index()

        if "clinician_action" in group.columns:
            clinician_counts = (
                group["clinician_action"]
                .replace(-1, np.nan)
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
            )
        else:
            clinician_counts = pd.Series(dtype=int)

        for action in range(25):
            fluid_bin, vasopressor_bin = decode_action(action)
            count = int(counts.get(action, 0))
            clinician_count = int(clinician_counts.get(action, 0))

            rows.append({
                "algorithm": algorithm,
                "seed": int(seed),
                "action": int(action),
                "fluid_bin": int(fluid_bin),
                "vasopressor_bin": int(vasopressor_bin),
                "count": count,
                "percentage": float(100.0 * count / total),
                "clinician_count": clinician_count,
                "clinician_percentage": float(100.0 * clinician_count / total),
            })

    return pd.DataFrame(rows)


def make_bin_summary_from_transition_df(action_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (algorithm, seed), group in action_df.groupby(["algorithm", "seed"]):
        rows.append({
            "algorithm": algorithm,
            "seed": int(seed),
            "n_transitions": int(len(group)),
            "mean_reward": float(group["reward"].mean()) if "reward" in group.columns else np.nan,
            "mean_fluid_bin": float(group["fluid_bin"].mean()),
            "mean_vasopressor_bin": float(group["vasopressor_bin"].mean()),
            "mean_clinician_fluid_bin": float(
                group["clinician_fluid_bin"].replace(-1, np.nan).mean()
            ) if "clinician_fluid_bin" in group.columns else np.nan,
            "mean_clinician_vasopressor_bin": float(
                group["clinician_vasopressor_bin"].replace(-1, np.nan).mean()
            ) if "clinician_vasopressor_bin" in group.columns else np.nan,
            "action_match_rate": float(group["action_match"].mean()) if "action_match" in group.columns else np.nan,
        })

    return pd.DataFrame(rows)


def make_bin_summary_from_action_summary(action_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (algorithm, seed), group in action_summary.groupby(["algorithm", "seed"]):
        weights = group["percentage"].astype(float).values

        if weights.sum() <= 0:
            continue

        clinician_weights = group["clinician_percentage"].astype(float).fillna(0).values

        rows.append({
            "algorithm": algorithm,
            "seed": int(seed),
            "n_transitions": int(group["count"].sum()),
            "mean_fluid_bin": float(np.average(group["fluid_bin"], weights=weights)),
            "mean_vasopressor_bin": float(np.average(group["vasopressor_bin"], weights=weights)),
            "mean_clinician_fluid_bin": float(np.average(group["fluid_bin"], weights=clinician_weights)) if clinician_weights.sum() > 0 else np.nan,
            "mean_clinician_vasopressor_bin": float(np.average(group["vasopressor_bin"], weights=clinician_weights)) if clinician_weights.sum() > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def make_heatmap_summary(action_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in action_summary.iterrows():
        rows.append({
            "algorithm": row["algorithm"],
            "seed": int(row["seed"]),
            "fluid_bin": int(row["fluid_bin"]),
            "vasopressor_bin": int(row["vasopressor_bin"]),
            "count": int(row["count"]),
            "percentage": float(row["percentage"]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CQL integration
# ---------------------------------------------------------------------

def find_cql_action_distribution_files() -> List[Path]:
    """
    Find CQL action-distribution files from evaluation seed folders only.

    This deliberately avoids cql_results_seed*/cql_action_distribution.csv
    to prevent duplicate CQL rows from training-output folders.
    """
    patterns = [
        str(SCRIPTS_DIR / "cql_evaluation_results_seed*" / "cql_eval_action_distribution.csv"),
        str(SCRIPTS_DIR / "cql_evaluation_results*" / "cql_eval_action_distribution.csv"),
        str(SCRIPT_DIR / "cql_evaluation_results_seed*" / "cql_eval_action_distribution.csv"),
        str(SCRIPTS_DIR / "results_5seed" / "**" / "cql_eval_action_distribution.csv"),
        str(SCRIPTS_DIR / "results" / "**" / "cql_eval_action_distribution.csv"),
    ]

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    unique_files = sorted(set(files))

    # Prefer clearly named seed folders if present.
    seed_files = [Path(f) for f in unique_files if re.search(r"cql_evaluation_results_seed\d+", f, flags=re.IGNORECASE)]
    if seed_files:
        return sorted(seed_files, key=lambda p: extract_seed_from_path(p))

    return [Path(file) for file in unique_files]


def load_cql_action_summary() -> pd.DataFrame:
    cql_files = find_cql_action_distribution_files()

    if not cql_files:
        print("[WARN] No CQL action distribution files found. Skipping CQL integration.")
        return pd.DataFrame()

    rows = []

    for cql_path in cql_files:
        seed = extract_seed_from_path(cql_path)
        print(f"[INFO] Loading CQL action distribution for seed {seed}: {cql_path}")

        cql_df = pd.read_csv(cql_path)

        for _, row in cql_df.iterrows():
            action = int(row["action"])
            fluid_bin = int(row.get("fluid_bin", action // 5))
            vasopressor_bin = int(row.get("vaso_bin", row.get("vasopressor_bin", action % 5)))

            cql_percent = row.get("cql_greedy_percent", row.get("percent", np.nan))
            dataset_percent = row.get("dataset_percent", np.nan)

            if pd.isna(cql_percent):
                continue

            rows.append({
                "algorithm": "CQL",
                "seed": int(seed),
                "action": action,
                "fluid_bin": fluid_bin,
                "vasopressor_bin": vasopressor_bin,
                "count": int(row.get("cql_greedy_count", row.get("count", 0))),
                "percentage": float(100.0 * cql_percent),
                "clinician_count": int(row.get("dataset_count", 0)),
                "clinician_percentage": float(100.0 * dataset_percent) if pd.notna(dataset_percent) else np.nan,
                "source_file": str(cql_path),
            })

    if not rows:
        return pd.DataFrame()

    cql_summary = pd.DataFrame(rows)

    # Keep only intended 5 seeds if detected.
    detected_seed_rows = cql_summary[cql_summary["seed"].isin(SEEDS)]
    if not detected_seed_rows.empty:
        cql_summary = detected_seed_rows.copy()

    # If accidental duplicates remain, aggregate by algorithm/seed/action.
    cql_summary = (
        cql_summary
        .groupby(["algorithm", "seed", "action", "fluid_bin", "vasopressor_bin"], as_index=False)
        .agg({
            "count": "sum",
            "percentage": "mean",
            "clinician_count": "sum",
            "clinician_percentage": "mean",
        })
    )

    return cql_summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_action_distribution(action_summary: pd.DataFrame, figures_dir: Path, timestamp: str) -> None:
    if action_summary.empty:
        return

    for algorithm in action_summary["algorithm"].unique():
        df_alg = action_summary[action_summary["algorithm"] == algorithm].copy()

        plt.figure(figsize=(10, 5))

        for seed, group in df_alg.groupby("seed"):
            group = group.sort_values("action")
            label = f"Seed {seed}" if int(seed) != -1 else "Seed unknown"
            plt.plot(group["action"], group["percentage"], marker="o", label=label)

        plt.xlabel("Action index")
        plt.ylabel("Selected actions (%)")
        plt.title(f"{algorithm} Selected Treatment Action Distribution")
        plt.xticks(range(25), rotation=90)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        path = figures_dir / f"{algorithm.lower()}_action_distribution_{timestamp}.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[INFO] Saved action distribution figure: {path}")


def plot_heatmaps(heatmap_df: pd.DataFrame, figures_dir: Path, timestamp: str) -> None:
    """
    Safe heatmap plotting.

    Uses aggregation + pivot_table to prevent:
        ValueError: Index contains duplicate entries, cannot reshape
    """
    if heatmap_df.empty:
        return

    for (algorithm, seed), group in heatmap_df.groupby(["algorithm", "seed"]):

        group = (
            group
            .groupby(["fluid_bin", "vasopressor_bin"], as_index=False)
            ["percentage"]
            .sum()
        )

        pivot = group.pivot_table(
            index="fluid_bin",
            columns="vasopressor_bin",
            values="percentage",
            aggfunc="sum",
            fill_value=0.0,
        )

        pivot = pivot.reindex(index=range(5), columns=range(5), fill_value=0.0)

        plt.figure(figsize=(7, 6))
        image = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(image, label="Selected actions (%)")
        plt.xlabel("Vasopressor intensity bin")
        plt.ylabel("Fluid intensity bin")

        seed_label = f"Seed {seed}" if int(seed) != -1 else "Seed unknown"
        plt.title(f"{algorithm} 25-Action Treatment Heatmap ({seed_label})")

        plt.xticks(range(5), range(5))
        plt.yticks(range(5), range(5))

        for i in range(5):
            for j in range(5):
                plt.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center")

        plt.tight_layout()

        seed_part = f"seed_{seed}" if int(seed) != -1 else "seed_unknown"
        path = figures_dir / f"{algorithm.lower()}_action_heatmap_{seed_part}_{timestamp}.png"

        plt.savefig(path, dpi=300)
        plt.close()

        print(f"[INFO] Saved heatmap figure: {path}")


def plot_bin_comparison(bin_summary: pd.DataFrame, figures_dir: Path, timestamp: str) -> None:
    if bin_summary.empty:
        return

    labels = [f"{row.algorithm}\nSeed {int(row.seed)}" for row in bin_summary.itertuples()]
    x = np.arange(len(bin_summary))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, bin_summary["mean_fluid_bin"], width=width, label="Fluid")
    plt.bar(x + width / 2, bin_summary["mean_vasopressor_bin"], width=width, label="Vasopressor")

    plt.xlabel("Algorithm / seed")
    plt.ylabel("Mean selected intensity bin")
    plt.title("Mean Fluid and Vasopressor Intensity by Policy")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    path = figures_dir / f"rl_mean_treatment_bins_{timestamp}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved bin comparison figure: {path}")


def plot_combined_action_distribution(combined_action_summary: pd.DataFrame, figures_dir: Path, timestamp: str) -> None:
    if combined_action_summary.empty:
        return

    summary = (
        combined_action_summary
        .groupby(["algorithm", "action"], as_index=False)["percentage"]
        .mean()
    )

    plt.figure(figsize=(11, 5))

    for algorithm, group in summary.groupby("algorithm"):
        group = group.sort_values("action")
        plt.plot(group["action"], group["percentage"], marker="o", label=algorithm)

    plt.xlabel("Action index")
    plt.ylabel("Selected actions (%)")
    plt.title("Average Treatment Action Distribution Across RL Baselines")
    plt.xticks(range(25), rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    path = figures_dir / f"combined_rl_action_distribution_{timestamp}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved combined action distribution figure: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PPO/A2C/CQL treatment action distributions.")

    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--figures_dir", type=str, default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--skip_cql", action="store_true")
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

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data path: {data_path}")
    print(f"[INFO] Model dir: {model_dir}")
    print(f"[INFO] Results dir: {results_dir}")
    print(f"[INFO] Figures dir: {figures_dir}")
    print(f"[INFO] Seeds: {SEEDS}")
    print(f"[INFO] Episodes per algorithm/seed: {args.n_episodes}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_action_dfs: List[pd.DataFrame] = []

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            try:
                df = collect_policy_actions(
                    algorithm=algorithm,
                    seed=seed,
                    data_path=data_path,
                    model_dir=model_dir,
                    device=device,
                    n_episodes=args.n_episodes,
                )
                all_action_dfs.append(df)
            except FileNotFoundError as exc:
                print(f"[WARN] Skipping {algorithm} Seed {seed}: {exc}")

    if not all_action_dfs:
        raise RuntimeError("No PPO/A2C action data were collected. Check model paths.")

    ppo_a2c_transition_df = pd.concat(all_action_dfs, ignore_index=True)
    transition_path = results_dir / f"ppo_a2c_temporal_action_distribution_{timestamp}.csv"
    ppo_a2c_transition_df.to_csv(transition_path, index=False)
    print(f"\n[INFO] Saved PPO/A2C transition-level action data: {transition_path}")

    ppo_a2c_action_summary = make_action_summary(ppo_a2c_transition_df)
    ppo_a2c_bin_summary = make_bin_summary_from_transition_df(ppo_a2c_transition_df)

    combined_action_summary = ppo_a2c_action_summary.copy()

    if not args.skip_cql:
        cql_action_summary = load_cql_action_summary()

        if not cql_action_summary.empty:
            cql_path = results_dir / f"cql_action_summary_integrated_{timestamp}.csv"
            cql_action_summary.to_csv(cql_path, index=False)
            print(f"[INFO] Saved integrated CQL action summary: {cql_path}")

            shared_cols = [
                "algorithm",
                "seed",
                "action",
                "fluid_bin",
                "vasopressor_bin",
                "count",
                "percentage",
                "clinician_count",
                "clinician_percentage",
            ]

            for col in shared_cols:
                if col not in cql_action_summary.columns:
                    cql_action_summary[col] = np.nan

            combined_action_summary = pd.concat(
                [combined_action_summary[shared_cols], cql_action_summary[shared_cols]],
                ignore_index=True,
            )
        else:
            print("[WARN] CQL action summary is empty. Combined table will include PPO/A2C only.")

    combined_heatmap_summary = make_heatmap_summary(combined_action_summary)
    combined_bin_summary = make_bin_summary_from_action_summary(combined_action_summary)

    ppo_a2c_action_summary_path = results_dir / f"ppo_a2c_temporal_action_summary_{timestamp}.csv"
    ppo_a2c_bin_summary_path = results_dir / f"ppo_a2c_temporal_bin_summary_{timestamp}.csv"
    combined_action_path = results_dir / f"combined_rl_action_summary_{timestamp}.csv"
    combined_bin_path = results_dir / f"combined_rl_temporal_bin_summary_{timestamp}.csv"
    combined_heatmap_path = results_dir / f"combined_rl_temporal_action_heatmap_summary_{timestamp}.csv"

    ppo_a2c_action_summary.to_csv(ppo_a2c_action_summary_path, index=False)
    ppo_a2c_bin_summary.to_csv(ppo_a2c_bin_summary_path, index=False)
    combined_action_summary.to_csv(combined_action_path, index=False)
    combined_bin_summary.to_csv(combined_bin_path, index=False)
    combined_heatmap_summary.to_csv(combined_heatmap_path, index=False)

    print(f"[INFO] Saved PPO/A2C action summary: {ppo_a2c_action_summary_path}")
    print(f"[INFO] Saved PPO/A2C bin summary: {ppo_a2c_bin_summary_path}")
    print(f"[INFO] Saved combined RL action summary: {combined_action_path}")
    print(f"[INFO] Saved combined RL bin summary: {combined_bin_path}")
    print(f"[INFO] Saved combined RL heatmap summary: {combined_heatmap_path}")

    print("\n=== PPO/A2C Bin Summary ===")
    print(ppo_a2c_bin_summary.to_string(index=False))

    print("\n=== Combined RL Bin Summary ===")
    print(combined_bin_summary.to_string(index=False))

    plot_action_distribution(combined_action_summary, figures_dir, timestamp)
    plot_heatmaps(combined_heatmap_summary, figures_dir, timestamp)
    plot_bin_comparison(combined_bin_summary, figures_dir, timestamp)
    plot_combined_action_distribution(combined_action_summary, figures_dir, timestamp)

    print("\n[DONE] Treatment-action analysis complete.")


if __name__ == "__main__":
    main()
