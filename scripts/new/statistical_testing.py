import os
import glob
import time
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu

# =========================================================
# CONFIG
# =========================================================

RESULTS_DIR = "../results"

timestamp = time.strftime("%Y%m%d_%H%M%S")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_csv_files(pattern, label):
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No {label} files found using pattern: {pattern}")

    print(f"\nLoaded {len(files)} {label} files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def get_reward_column(df):
    possible_cols = [
        "total_reward",
        "episode_reward",
        "mean_reward",
        "reward"
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise KeyError(
        f"No reward column found. Available columns: {df.columns.tolist()}"
    )


def mann_whitney_test(group_a, group_b, comparison_name):
    group_a = pd.Series(group_a).dropna()
    group_b = pd.Series(group_b).dropna()

    if len(group_a) == 0 or len(group_b) == 0:
        return {
            "comparison": comparison_name,
            "n_group_a": len(group_a),
            "n_group_b": len(group_b),
            "u_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "interpretation": "Insufficient data"
        }

    u_stat, p_value = mannwhitneyu(
        group_a,
        group_b,
        alternative="two-sided"
    )

    return {
        "comparison": comparison_name,
        "n_group_a": len(group_a),
        "n_group_b": len(group_b),
        "u_statistic": u_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "interpretation": (
            "Statistically significant"
            if p_value < 0.05
            else "Not statistically significant"
        )
    }


# =========================================================
# LOAD PPO AND A2C EPISODE REWARD FILES
# =========================================================

ppo_pattern = f"{RESULTS_DIR}/ppo_temporal_25_seed_*_episode_rewards_*.csv"
a2c_pattern = f"{RESULTS_DIR}/a2c_temporal_25_seed_*_episode_rewards_*.csv"

ppo_df = load_csv_files(ppo_pattern, "PPO episode reward")
a2c_df = load_csv_files(a2c_pattern, "A2C episode reward")

ppo_reward_col = get_reward_column(ppo_df)
a2c_reward_col = get_reward_column(a2c_df)

print("\nDetected reward columns:")
print(f"PPO reward column: {ppo_reward_col}")
print(f"A2C reward column: {a2c_reward_col}")

ppo_rewards = ppo_df[ppo_reward_col]
a2c_rewards = a2c_df[a2c_reward_col]

# =========================================================
# PPO VS A2C STATISTICAL TEST
# =========================================================

stats_results = []

ppo_vs_a2c = mann_whitney_test(
    ppo_rewards,
    a2c_rewards,
    "PPO_vs_A2C_episode_rewards"
)

stats_results.append(ppo_vs_a2c)

print("\n================================================")
print("PPO vs A2C Mann-Whitney U Test")
print("================================================")
print(f"N PPO       : {ppo_vs_a2c['n_group_a']}")
print(f"N A2C       : {ppo_vs_a2c['n_group_b']}")
print(f"U-statistic : {ppo_vs_a2c['u_statistic']:.4f}")
print(f"P-value     : {ppo_vs_a2c['p_value']:.8f}")
print(f"Result      : {ppo_vs_a2c['interpretation']}")

# =========================================================
# PPO SEED-TO-SEED COMPARISONS
# =========================================================

if "seed" in ppo_df.columns:
    print("\n================================================")
    print("PPO Seed-to-Seed Mann-Whitney U Tests")
    print("================================================")

    available_seeds = sorted(ppo_df["seed"].dropna().unique())

    for i in range(len(available_seeds)):
        for j in range(i + 1, len(available_seeds)):
            seed_a = available_seeds[i]
            seed_b = available_seeds[j]

            rewards_a = ppo_df[ppo_df["seed"] == seed_a][ppo_reward_col]
            rewards_b = ppo_df[ppo_df["seed"] == seed_b][ppo_reward_col]

            comparison_name = f"PPO_seed_{seed_a}_vs_seed_{seed_b}"

            result = mann_whitney_test(
                rewards_a,
                rewards_b,
                comparison_name
            )

            stats_results.append(result)

            print(f"\nSeed {seed_a} vs Seed {seed_b}")
            print(f"N Seed {seed_a}: {result['n_group_a']}")
            print(f"N Seed {seed_b}: {result['n_group_b']}")
            print(f"U-statistic : {result['u_statistic']:.4f}")
            print(f"P-value     : {result['p_value']:.8f}")
            print(f"Result      : {result['interpretation']}")

# =========================================================
# SEVERITY SUBGROUP STATISTICAL TESTS
# =========================================================

severity_pattern = f"{RESULTS_DIR}/severity_subgroup_rollouts_*.csv"
severity_files = glob.glob(severity_pattern)

if len(severity_files) > 0:
    severity_df = load_csv_files(
        severity_pattern,
        "severity subgroup rollout"
    )

    if "severity_group" in severity_df.columns:

        severity_df = severity_df[
            severity_df["severity_group"] != "Unknown"
        ].copy()

        severity_reward_col = get_reward_column(severity_df)

        print("\nDetected severity reward column:")
        print(f"Severity reward column: {severity_reward_col}")

        comparisons = [
            ("Low", "Medium"),
            ("Medium", "High"),
            ("Low", "High")
        ]

        print("\n================================================")
        print("Severity Subgroup Mann-Whitney U Tests")
        print("================================================")

        for group_a, group_b in comparisons:

            rewards_a = severity_df[
                severity_df["severity_group"] == group_a
            ][severity_reward_col]

            rewards_b = severity_df[
                severity_df["severity_group"] == group_b
            ][severity_reward_col]

            comparison_name = f"Severity_{group_a}_vs_{group_b}"

            result = mann_whitney_test(
                rewards_a,
                rewards_b,
                comparison_name
            )

            stats_results.append(result)

            print(f"\n{group_a} vs {group_b}")
            print(f"N {group_a}: {result['n_group_a']}")
            print(f"N {group_b}: {result['n_group_b']}")
            print(f"U-statistic : {result['u_statistic']:.4f}")
            print(f"P-value     : {result['p_value']:.8f}")
            print(f"Result      : {result['interpretation']}")

    else:
        print("\nSeverity files found, but no severity_group column detected.")
else:
    print("\nNo severity subgroup rollout files found. Skipping severity tests.")

# =========================================================
# SAVE STATISTICAL RESULTS
# =========================================================

stats_df = pd.DataFrame(stats_results)

save_path = f"{RESULTS_DIR}/statistical_testing_results_{timestamp}.csv"
stats_df.to_csv(save_path, index=False)

latest_path = f"{RESULTS_DIR}/statistical_testing_results_latest.csv"
stats_df.to_csv(latest_path, index=False)

print("\n================================================")
print("Saved statistical testing results to:")
print(save_path)
print(latest_path)
print("================================================")

print("\nFinal Statistical Testing Summary:")
print(stats_df)