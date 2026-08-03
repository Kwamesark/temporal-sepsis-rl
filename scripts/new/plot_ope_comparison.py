import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "../results"
FIGURES_DIR = "../figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================================================
# Load only PPO temporal seeds 3,4,5
# =========================================================

ope_files = glob.glob(f"{RESULTS_DIR}/ppo_temporal_25_seed_*_ope_metrics_*.csv")

if len(ope_files) == 0:
    raise FileNotFoundError("No PPO temporal OPE metric files found.")

dfs = []

for file in ope_files:
    df = pd.read_csv(file)

    # Keep only seeds 3,4,5
    if int(df["seed"].iloc[0]) in [3, 4, 5]:
        dfs.append(df)

ope_df = pd.concat(dfs, ignore_index=True)

# Sort seeds
ope_df = ope_df.sort_values("seed")

print("\n=== PPO Temporal OPE Results ===")
print(ope_df[[
    "seed",
    "true_return",
    "is_estimate",
    "wis_estimate",
    "dm_estimate"
]])

# =========================================================
# Save combined CSV
# =========================================================

combined_path = f"{RESULTS_DIR}/combined_ppo_temporal_ope_metrics.csv"
ope_df.to_csv(combined_path, index=False)

print(f"\nSaved combined OPE table to:\n{combined_path}")

# =========================================================
# Figure 1: Stable OPE Comparison
# =========================================================

plot_df = ope_df[[
    "seed",
    "true_return",
    "wis_estimate",
    "dm_estimate"
]].copy()

plot_df = plot_df.set_index("seed")

ax = plot_df.plot(
    kind="bar",
    figsize=(9,5)
)

plt.xlabel("Seed")
plt.ylabel("Estimated Return")
plt.title("OPE Comparison Across PPO Temporal Seeds")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()

stable_path = f"{FIGURES_DIR}/ope_comparison.png"

plt.savefig(stable_path, dpi=300)

plt.close()

print(f"\nSaved stable OPE figure to:\n{stable_path}")

# =========================================================
# Figure 2: IS Instability
# =========================================================

is_df = ope_df[["seed", "is_estimate"]].copy()

is_df["abs_is_estimate"] = is_df["is_estimate"].abs()

plt.figure(figsize=(8,5))

plt.bar(
    is_df["seed"].astype(str),
    is_df["abs_is_estimate"]
)

plt.yscale("log")

plt.xlabel("Seed")
plt.ylabel("Absolute IS Estimate (log scale)")
plt.title("Importance Sampling Instability")

plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()

is_path = f"{FIGURES_DIR}/is_instability.png"

plt.savefig(is_path, dpi=300)

plt.close()

print(f"\nSaved IS instability figure to:\n{is_path}")