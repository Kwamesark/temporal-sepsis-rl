#!/usr/bin/env python3
"""
prepare_offline_dataset.py

Prepare an offline reinforcement learning dataset for Conservative Q-Learning (CQL)
using temporal ICU trajectories for sepsis treatment optimization.

The script converts temporally ordered patient ICU trajectories into transition tuples:

    (state_t, action_t, reward_t, next_state_t, done_t)

Each ICU stay is treated as one episode.
Each 4-hour clinical window is treated as one timestep.

Expected input:
    A CSV file where each row represents one 4-hour ICU window.

Required minimum columns:
    stay_id
    window_index or charttime
    clinical state variables
    fluid amount or fluid bin
    vasopressor amount or vasopressor bin

Optional columns:
    lactate
    map
    sofa_proxy
    mortality
    hospital_expire_flag
    action

Outputs:
    offline_cql_dataset.npz
    offline_cql_transitions.csv
    offline_cql_metadata.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Default feature columns used in your temporal sepsis RL framework
# ---------------------------------------------------------------------

DEFAULT_STATE_COLUMNS = [
    "heart_rate",
    "map",
    "resp_rate",
    "spo2",
    "temperature",
    "lactate",
    "creatinine",
    "wbc",
    "platelet",
    "bilirubin",
    "iv_fluid",
    "vasopressor_rate",
    "age",
    "gender",
    "sofa_proxy",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def find_existing_columns(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    """
    Return only columns that exist in the dataframe.
    """
    return [col for col in candidate_cols if col in df.columns]


def require_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Raise a clear error if required columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns are: {list(df.columns)}"
        )


def safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Convert a pandas Series to numeric and fill missing values.
    """
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def get_sort_column(df: pd.DataFrame) -> str:
    """
    Determine the best temporal ordering column.
    """
    possible_time_cols = [
        "window_index",
        "time_bin",
        "timestep",
        "charttime",
        "starttime",
        "window_start",
        "window_end",
    ]

    for col in possible_time_cols:
        if col in df.columns:
            return col

    raise ValueError(
        "No temporal ordering column found. Please include one of: "
        "window_index, time_bin, timestep, charttime, starttime, window_start, window_end."
    )


def identify_mortality_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify mortality/outcome column if available.
    """
    possible_cols = [
        "mortality",
        "hospital_expire_flag",
        "in_hospital_mortality",
        "death",
        "died",
        "expire_flag",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    return None


# ---------------------------------------------------------------------
# Action construction
# ---------------------------------------------------------------------

def create_bins_from_quantiles(
    values: pd.Series,
    n_bins: int = 5,
    zero_bin: bool = True
) -> pd.Series:
    """
    Convert continuous treatment values into ordinal bins.

    If zero_bin=True:
        value <= 0 is assigned to bin 0.
        positive values are split into bins 1 to n_bins - 1 using quantiles.

    This is useful for fluids and vasopressors where no treatment should be bin 0.
    """

    values = safe_numeric(values, fill_value=0.0)

    bins = pd.Series(np.zeros(len(values), dtype=int), index=values.index)

    if zero_bin:
        positive_mask = values > 0
        positive_values = values[positive_mask]

        if len(positive_values) == 0:
            return bins

        try:
            positive_bins = pd.qcut(
                positive_values,
                q=n_bins - 1,
                labels=False,
                duplicates="drop"
            )

            positive_bins = positive_bins.astype(int) + 1
            bins.loc[positive_mask] = positive_bins

        except ValueError:
            bins.loc[positive_mask] = 1

    else:
        try:
            bins = pd.qcut(
                values,
                q=n_bins,
                labels=False,
                duplicates="drop"
            ).astype(int)
        except ValueError:
            bins = pd.Series(np.zeros(len(values), dtype=int), index=values.index)

    bins = bins.clip(lower=0, upper=n_bins - 1).astype(int)

    return bins


def build_25_action_space(
    df: pd.DataFrame,
    fluid_col: str,
    vaso_col: str,
    fluid_bin_col: Optional[str] = None,
    vaso_bin_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Build 25-action treatment representation:

        action = 5 * fluid_bin + vaso_bin

    fluid_bin in {0,1,2,3,4}
    vaso_bin in {0,1,2,3,4}
    action in {0,...,24}
    """

    df = df.copy()

    if "action" in df.columns:
        df["action"] = safe_numeric(df["action"], fill_value=0).astype(int)
        df["action"] = df["action"].clip(lower=0, upper=24)
        return df

    if fluid_bin_col and fluid_bin_col in df.columns:
        df["fluid_bin"] = safe_numeric(df[fluid_bin_col], fill_value=0).astype(int)
    else:
        df["fluid_bin"] = create_bins_from_quantiles(df[fluid_col], n_bins=5, zero_bin=True)

    if vaso_bin_col and vaso_bin_col in df.columns:
        df["vaso_bin"] = safe_numeric(df[vaso_bin_col], fill_value=0).astype(int)
    else:
        df["vaso_bin"] = create_bins_from_quantiles(df[vaso_col], n_bins=5, zero_bin=True)

    df["fluid_bin"] = df["fluid_bin"].clip(lower=0, upper=4).astype(int)
    df["vaso_bin"] = df["vaso_bin"].clip(lower=0, upper=4).astype(int)

    df["action"] = 5 * df["fluid_bin"] + df["vaso_bin"]
    df["action"] = df["action"].clip(lower=0, upper=24).astype(int)

    return df


# ---------------------------------------------------------------------
# Reward engineering
# ---------------------------------------------------------------------

def compute_step_reward(
    current_row: pd.Series,
    next_row: pd.Series,
    mortality_col: Optional[str],
    terminal: bool,
    lactate_col: str = "lactate",
    map_col: str = "map",
    sofa_col: str = "sofa_proxy",
    fluid_bin_col: str = "fluid_bin",
    vaso_bin_col: str = "vaso_bin"
) -> float:
    """
    Clinically informed reward function.

    Reward components:
        + lactate improvement
        + MAP stabilization
        + SOFA-like severity reduction
        - treatment intensity penalty
        - mortality penalty at terminal state

    This is intentionally conservative and interpretable.
    """

    reward = 0.0

    # Lactate improvement
    if lactate_col in current_row.index and lactate_col in next_row.index:
        lactate_t = current_row[lactate_col]
        lactate_tp1 = next_row[lactate_col]

        if pd.notna(lactate_t) and pd.notna(lactate_tp1):
            reward += 0.20 * np.tanh(lactate_t - lactate_tp1)

    # MAP stabilization
    if map_col in next_row.index:
        map_next = next_row[map_col]

        if pd.notna(map_next):
            if 65 <= map_next <= 100:
                reward += 0.20
            elif map_next < 65:
                reward -= 0.20
            elif map_next > 120:
                reward -= 0.10

    # SOFA-like severity reduction
    if sofa_col in current_row.index and sofa_col in next_row.index:
        sofa_t = current_row[sofa_col]
        sofa_tp1 = next_row[sofa_col]

        if pd.notna(sofa_t) and pd.notna(sofa_tp1):
            reward += 0.20 * np.tanh(sofa_t - sofa_tp1)

    # Treatment intensity penalty
    fluid_bin = current_row.get(fluid_bin_col, 0)
    vaso_bin = current_row.get(vaso_bin_col, 0)

    treatment_penalty = 0.02 * float(fluid_bin) + 0.03 * float(vaso_bin)
    reward -= treatment_penalty

    # Terminal mortality penalty or survival reward
    if terminal and mortality_col is not None:
        mortality_value = current_row.get(mortality_col, 0)

        try:
            mortality_value = int(mortality_value)
        except Exception:
            mortality_value = 0

        if mortality_value == 1:
            reward -= 1.0
        else:
            reward += 0.5

    return float(reward)


# ---------------------------------------------------------------------
# Offline transition construction
# ---------------------------------------------------------------------

def normalize_features(
    df: pd.DataFrame,
    state_columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Z-score normalize state features.

    Returns:
        normalized dataframe
        normalization statistics
    """

    df = df.copy()
    stats = {}

    for col in state_columns:
        values = safe_numeric(df[col], fill_value=np.nan)

        mean = float(values.mean(skipna=True))
        std = float(values.std(skipna=True))

        if np.isnan(mean):
            mean = 0.0

        if np.isnan(std) or std == 0:
            std = 1.0

        df[col] = values.fillna(mean)
        df[col] = (df[col] - mean) / std

        stats[col] = {
            "mean": mean,
            "std": std
        }

    return df, stats


def build_transitions(
    df: pd.DataFrame,
    state_columns: List[str],
    stay_id_col: str,
    sort_col: str,
    mortality_col: Optional[str],
    normalize: bool = True
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict]:
    """
    Build offline RL transitions from temporal ICU trajectories.
    """

    df = df.copy()

    # Sort by stay and time
    df = df.sort_values([stay_id_col, sort_col]).reset_index(drop=True)

    # Keep only useful state columns that exist
    state_columns = find_existing_columns(df, state_columns)

    if len(state_columns) == 0:
        raise ValueError("No valid state columns found. Check your column names.")

    # Normalize state features
    normalization_stats = {}
    if normalize:
        df, normalization_stats = normalize_features(df, state_columns)
    else:
        for col in state_columns:
            df[col] = safe_numeric(df[col], fill_value=0.0)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    stay_ids = []
    timesteps = []

    transition_rows = []

    episode_lengths = []

    grouped = df.groupby(stay_id_col, sort=False)

    for stay_id, group in grouped:
        group = group.sort_values(sort_col).reset_index(drop=True)

        if len(group) < 2:
            continue

        episode_lengths.append(len(group))

        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i + 1]

            terminal = i == len(group) - 2

            state = current_row[state_columns].to_numpy(dtype=np.float32)
            next_state = next_row[state_columns].to_numpy(dtype=np.float32)

            action = int(current_row["action"])
            reward = compute_step_reward(
                current_row=current_row,
                next_row=next_row,
                mortality_col=mortality_col,
                terminal=terminal
            )

            done = bool(terminal)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            stay_ids.append(stay_id)
            timesteps.append(i)

            transition_rows.append({
                "stay_id": stay_id,
                "timestep": i,
                "action": action,
                "reward": reward,
                "done": int(done),
                "fluid_bin": int(current_row.get("fluid_bin", action // 5)),
                "vaso_bin": int(current_row.get("vaso_bin", action % 5)),
            })

    if len(states) == 0:
        raise ValueError(
            "No transitions were created. Make sure each stay_id has at least two 4-hour windows."
        )

    dataset = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "stay_ids": np.asarray(stay_ids),
        "timesteps": np.asarray(timesteps, dtype=np.int64),
    }

    transitions_df = pd.DataFrame(transition_rows)

    metadata = {
        "num_transitions": int(len(states)),
        "num_episodes": int(len(episode_lengths)),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "min_episode_length": int(np.min(episode_lengths)) if episode_lengths else 0,
        "max_episode_length": int(np.max(episode_lengths)) if episode_lengths else 0,
        "state_dim": int(len(state_columns)),
        "num_actions": 25,
        "state_columns": state_columns,
        "stay_id_col": stay_id_col,
        "sort_col": sort_col,
        "mortality_col": mortality_col,
        "normalization_used": normalize,
        "normalization_stats": normalization_stats,
        "reward_description": {
            "lactate": "positive reward for lactate reduction",
            "map": "positive reward for MAP between 65 and 100",
            "sofa_proxy": "positive reward for SOFA-like severity reduction",
            "treatment_penalty": "small penalty for higher fluid and vasopressor intensity",
            "mortality": "terminal penalty for death and terminal reward for survival"
        }
    }

    return dataset, transitions_df, metadata


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare offline CQL dataset from temporal sepsis ICU trajectories."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to temporal ICU trajectory CSV file."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="offline_dataset",
        help="Directory where offline dataset files will be saved."
    )

    parser.add_argument(
        "--stay_id_col",
        type=str,
        default="stay_id",
        help="Column identifying each ICU stay."
    )

    parser.add_argument(
        "--fluid_col",
        type=str,
        default="iv_fluid",
        help="Column containing IV fluid amount or intensity."
    )

    parser.add_argument(
        "--vaso_col",
        type=str,
        default="vasopressor_rate",
        help="Column containing vasopressor dose/rate or intensity."
    )

    parser.add_argument(
        "--fluid_bin_col",
        type=str,
        default=None,
        help="Optional existing fluid bin column."
    )

    parser.add_argument(
        "--vaso_bin_col",
        type=str,
        default=None,
        help="Optional existing vasopressor bin column."
    )

    parser.add_argument(
        "--state_cols",
        type=str,
        default=None,
        help="Optional comma-separated state columns. If not provided, default clinical columns are used."
    )

    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable z-score normalization of state features."
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Loading temporal ICU trajectory data from: {input_path}")
    df = pd.read_csv(input_path)

    require_columns(df, [args.stay_id_col])

    sort_col = get_sort_column(df)
    mortality_col = identify_mortality_column(df)

    if args.state_cols:
        state_columns = [col.strip() for col in args.state_cols.split(",")]
    else:
        state_columns = DEFAULT_STATE_COLUMNS

    state_columns = find_existing_columns(df, state_columns)

    if len(state_columns) == 0:
        raise ValueError(
            "No state columns found. Provide columns using --state_cols."
        )

    # Make sure fluid and vasopressor columns exist unless action already exists
    if "action" not in df.columns:
        require_columns(df, [args.fluid_col, args.vaso_col])

    print("[INFO] Building 25-action fluid-vasopressor treatment space...")
    df = build_25_action_space(
        df=df,
        fluid_col=args.fluid_col,
        vaso_col=args.vaso_col,
        fluid_bin_col=args.fluid_bin_col,
        vaso_bin_col=args.vaso_bin_col
    )

    print("[INFO] Constructing offline transition tuples...")
    dataset, transitions_df, metadata = build_transitions(
        df=df,
        state_columns=state_columns,
        stay_id_col=args.stay_id_col,
        sort_col=sort_col,
        mortality_col=mortality_col,
        normalize=not args.no_normalize
    )

    npz_path = output_dir / "offline_cql_dataset.npz"
    csv_path = output_dir / "offline_cql_transitions.csv"
    metadata_path = output_dir / "offline_cql_metadata.json"

    print(f"[INFO] Saving NPZ dataset to: {npz_path}")
    np.savez_compressed(
        npz_path,
        states=dataset["states"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        next_states=dataset["next_states"],
        dones=dataset["dones"],
        stay_ids=dataset["stay_ids"],
        timesteps=dataset["timesteps"]
    )

    print(f"[INFO] Saving transition summary CSV to: {csv_path}")
    transitions_df.to_csv(csv_path, index=False)

    print(f"[INFO] Saving metadata to: {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n[DONE] Offline CQL dataset prepared successfully.")
    print(f"Number of transitions: {metadata['num_transitions']}")
    print(f"Number of episodes: {metadata['num_episodes']}")
    print(f"State dimension: {metadata['state_dim']}")
    print(f"Action space size: {metadata['num_actions']}")
    print(f"Mean episode length: {metadata['mean_episode_length']:.2f}")
    print(f"Mortality column used: {metadata['mortality_col']}")
    print(f"Temporal sort column used: {metadata['sort_col']}")
    print("\nOutput files:")
    print(f"  - {npz_path}")
    print(f"  - {csv_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()