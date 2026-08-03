import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.new.sepsis_temporal_env import SepsisTrajectoryEnv

# =========================================================
# CONFIG
# =========================================================

SEEDS = [3, 4, 5]

MODEL_DIR = "../models"
DATA_PATH = "../data/sepsis_trajectories.csv"
RESULTS_DIR = "../results"
FIGURES_DIR = "../figures"

N_EPISODES = 200

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

# =========================================================
# CREATE SOFA-LIKE SEVERITY SCORE IF MISSING
# =========================================================

if "sofa_proxy" not in df.columns:
    print("sofa_proxy column not found. Computing sofa_proxy manually...")

    def compute_sofa_proxy(row):
        score = 0

        platelets = row.get("platelets", np.nan)
        bilirubin = row.get("bilirubin", np.nan)
        map_value = row.get("mean_arterial_pressure", np.nan)
        creatinine = row.get("creatinine", np.nan)
        spo2 = row.get("spo2", np.nan)

        if pd.notna(platelets):
            if platelets < 50:
                score += 4
            elif platelets < 100:
                score += 3
            elif platelets < 150:
                score += 2
            elif platelets < 200:
                score += 1

        if pd.notna(bilirubin):
            if bilirubin >= 12:
                score += 4
            elif bilirubin >= 6:
                score += 3
            elif bilirubin >= 2:
                score += 2
            elif bilirubin >= 1.2:
                score += 1

        if pd.notna(map_value):
            if map_value < 50:
                score += 3
            elif map_value < 65:
                score += 2
            elif map_value < 70:
                score += 1

        if pd.notna(creatinine):
            if creatinine >= 5:
                score += 4
            elif creatinine >= 3.5:
                score += 3
            elif creatinine >= 2:
                score += 2
            elif creatinine >= 1.2:
                score += 1

        if pd.notna(spo2):
            if spo2 < 85:
                score += 3
            elif spo2 < 90:
                score += 2
            elif spo2 < 94:
                score += 1

        return score

    df["sofa_proxy"] = df.apply(compute_sofa_proxy, axis=1)

severity_col = "sofa_proxy"

# =========================================================
# CREATE SEVERITY GROUPS
# =========================================================

low_threshold = df[severity_col].quantile(0.33)
high_threshold = df[severity_col].quantile(0.66)

print("\nSeverity Thresholds")
print("-------------------")
print(f"Low threshold  : {low_threshold:.2f}")
print(f"High threshold : {high_threshold:.2f}")

def classify_severity(x):
    if x <= low_threshold:
        return "Low"
    elif x <= high_threshold:
        return "Medium"
    else:
        return "High"

df["severity_group"] = df[severity_col].apply(classify_severity)

print("\nSeverity Group Counts")
print(df["severity_group"].value_counts())

# Map stay_id to baseline severity group using mean severity per ICU stay
stay_severity = (
    df.groupby("stay_id")[severity_col]
    .mean()
    .reset_index()
)

stay_severity["severity_group"] = stay_severity[severity_col].apply(classify_severity)

stay_to_severity = dict(
    zip(stay_severity["stay_id"], stay_severity["severity_group"])
)

# =========================================================
# PPO ROLLOUTS BY SEVERITY
# =========================================================

rollout_records = []

for seed in SEEDS:
    print(f"\nRunning severity subgroup rollout for PPO seed {seed}...")

    model_path = f"{MODEL_DIR}/ppo_temporal_25_seed_{seed}"
    vec_path = f"{MODEL_DIR}/ppo_temporal_25_seed_{seed}_vecnormalize.pkl"

    raw_env = DummyVecEnv([
        lambda: SepsisTrajectoryEnv(data_path=DATA_PATH)
    ])

    env = VecNormalize.load(vec_path, raw_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device=device)

    for episode in range(N_EPISODES):
        obs = env.reset()
        done = False

        total_reward = 0.0
        episode_length = 0
        selected_actions = []

        # Get stay_id from underlying environment if available
        try:
            current_stay_id = int(env.venv.envs[0].current_stay_id)
        except Exception:
            current_stay_id = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            action_int = int(action[0]) if isinstance(action, np.ndarray) else int(action)

            obs, reward, done, info = env.step(action)

            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)

            total_reward += reward_value
            episode_length += 1
            selected_actions.append(action_int)

        if current_stay_id is not None and current_stay_id in stay_to_severity:
            severity_group = stay_to_severity[current_stay_id]
        else:
            severity_group = "Unknown"

        mean_action = np.mean(selected_actions) if len(selected_actions) > 0 else np.nan
        mean_fluid_bin = np.mean([a // 5 for a in selected_actions]) if len(selected_actions) > 0 else np.nan
        mean_vaso_bin = np.mean([a % 5 for a in selected_actions]) if len(selected_actions) > 0 else np.nan

        rollout_records.append({
            "seed": seed,
            "episode": episode + 1,
            "stay_id": current_stay_id,
            "severity_group": severity_group,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "mean_action": mean_action,
            "mean_fluid_bin": mean_fluid_bin,
            "mean_vasopressor_bin": mean_vaso_bin,
        })

    env.close()

# =========================================================
# SAVE ROLLOUT DATA
# =========================================================

rollout_df = pd.DataFrame(rollout_records)

rollout_path = f"{RESULTS_DIR}/severity_subgroup_rollouts_{timestamp}.csv"
rollout_df.to_csv(rollout_path, index=False)

print(f"\nSaved severity rollout data to {rollout_path}")

# Remove unknown if any
analysis_df = rollout_df[rollout_df["severity_group"] != "Unknown"].copy()

# =========================================================
# SUMMARY TABLE
# =========================================================

summary_df = (
    analysis_df
    .groupby("severity_group")
    .agg(
        mean_reward=("total_reward", "mean"),
        std_reward=("total_reward", "std"),
        min_reward=("total_reward", "min"),
        max_reward=("total_reward", "max"),
        mean_episode_length=("episode_length", "mean"),
        mean_fluid_bin=("mean_fluid_bin", "mean"),
        mean_vasopressor_bin=("mean_vasopressor_bin", "mean"),
        n_episodes=("episode", "count")
    )
    .reset_index()
)

severity_order = ["Low", "Medium", "High"]
summary_df["severity_group"] = pd.Categorical(
    summary_df["severity_group"],
    categories=severity_order,
    ordered=True
)
summary_df = summary_df.sort_values("severity_group")

summary_path = f"{RESULTS_DIR}/severity_subgroup_summary_{timestamp}.csv"
summary_df.to_csv(summary_path, index=False)

print("\nSeverity Subgroup Summary")
print(summary_df)

print(f"\nSaved severity summary to {summary_path}")

# =========================================================
# FIGURE 1: MEAN REWARD BY SEVERITY
# =========================================================

plt.figure(figsize=(7, 5))
plt.bar(summary_df["severity_group"].astype(str), summary_df["mean_reward"])

plt.xlabel("Severity Group")
plt.ylabel("Mean Episode Reward")
plt.title("PPO Mean Reward by Patient Severity")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

reward_fig = f"{FIGURES_DIR}/severity_reward_comparison.png"
plt.savefig(reward_fig, dpi=300)
plt.close()

print(f"Saved severity reward figure to {reward_fig}")

# =========================================================
# FIGURE 2: REWARD DISTRIBUTION BY SEVERITY
# =========================================================

plt.figure(figsize=(8, 5))

groups = [
    analysis_df[analysis_df["severity_group"] == group]["total_reward"].values
    for group in severity_order
]

plt.boxplot(groups, labels=severity_order, showmeans=True)

plt.xlabel("Severity Group")
plt.ylabel("Episode Reward")
plt.title("PPO Reward Distribution by Patient Severity")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

box_fig = f"{FIGURES_DIR}/severity_reward_boxplot.png"
plt.savefig(box_fig, dpi=300)
plt.close()

print(f"Saved severity reward boxplot to {box_fig}")

# =========================================================
# FIGURE 3: MEAN EPISODE LENGTH BY SEVERITY
# =========================================================

plt.figure(figsize=(7, 5))
plt.bar(summary_df["severity_group"].astype(str), summary_df["mean_episode_length"])

plt.xlabel("Severity Group")
plt.ylabel("Mean Episode Length")
plt.title("PPO Mean Episode Length by Patient Severity")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

length_fig = f"{FIGURES_DIR}/severity_episode_length.png"
plt.savefig(length_fig, dpi=300)
plt.close()

print(f"Saved severity episode length figure to {length_fig}")

# =========================================================
# FIGURE 4: MEAN FLUID BIN BY SEVERITY
# =========================================================

plt.figure(figsize=(7, 5))
plt.bar(summary_df["severity_group"].astype(str), summary_df["mean_fluid_bin"])

plt.xlabel("Severity Group")
plt.ylabel("Mean Fluid Intensity Bin")
plt.title("Mean PPO Fluid Intensity by Patient Severity")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

fluid_fig = f"{FIGURES_DIR}/severity_fluid_bin.png"
plt.savefig(fluid_fig, dpi=300)
plt.close()

print(f"Saved severity fluid bin figure to {fluid_fig}")

# =========================================================
# FIGURE 5: MEAN VASOPRESSOR BIN BY SEVERITY
# =========================================================

plt.figure(figsize=(7, 5))
plt.bar(summary_df["severity_group"].astype(str), summary_df["mean_vasopressor_bin"])

plt.xlabel("Severity Group")
plt.ylabel("Mean Vasopressor Intensity Bin")
plt.title("Mean PPO Vasopressor Intensity by Patient Severity")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

vaso_fig = f"{FIGURES_DIR}/severity_vasopressor_bin.png"
plt.savefig(vaso_fig, dpi=300)
plt.close()

print(f"Saved severity vasopressor bin figure to {vaso_fig}")

print("\nSeverity subgroup analysis complete.")