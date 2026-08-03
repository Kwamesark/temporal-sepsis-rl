#!/usr/bin/env python3
"""
extract_trajectories.py

Extract temporal sepsis ICU trajectories from MIMIC-IV v3.1 using BigQuery
and prepare the CSV used by:

    - sepsis_temporal_env.py
    - train_ppo_temporal_25.py / train_rl_agent.py
    - train_a2c_agent.py
    - prepare_offline_dataset_env_aligned.py
    - train_cql_agent.py
    - evaluate_cql_agent.py

Output:
    ../data/sepsis_trajectories.csv

Trajectory design:
    - One ICU stay = one episode
    - One row = one 4-hour ICU window
    - Action space = 25 treatment actions
    - action = fluid_bin * 5 + vasopressor_bin

Important action-binning rule:
    - No fluid or no vasopressor treatment is always bin 0
    - Positive treatment amounts are split into bins 1-4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR.parent / "data" / "sepsis_trajectories.csv"


# ---------------------------------------------------------------------
# BigQuery SQL
# ---------------------------------------------------------------------

def build_query(max_timestep: int = 20, row_limit: int = 200000) -> str:
    """
    Build MIMIC-IV v3.1 temporal trajectory extraction query.
    """

    query = f"""
WITH eligible_icustays AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        intime,
        outtime
    FROM `physionet-data.mimiciv_3_1_icu.icustays`
    WHERE DATETIME_DIFF(outtime, intime, HOUR) >= 24
),

infection_flags AS (
    SELECT DISTINCT
        p.hadm_id
    FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` p
    JOIN `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
        ON p.hadm_id = m.hadm_id
    WHERE p.drug IS NOT NULL
      AND m.spec_type_desc IS NOT NULL
),

time_windows AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        window_start,
        TIMESTAMP_ADD(window_start, INTERVAL 4 HOUR) AS window_end,
        DIV(TIMESTAMP_DIFF(window_start, TIMESTAMP(i.intime), HOUR), 4) AS timestep
    FROM eligible_icustays i,
    UNNEST(
        GENERATE_TIMESTAMP_ARRAY(
            TIMESTAMP(i.intime),
            TIMESTAMP(DATETIME_SUB(i.outtime, INTERVAL 4 HOUR)),
            INTERVAL 4 HOUR
        )
    ) AS window_start
),

vitals AS (
    SELECT
        tw.stay_id,
        tw.timestep,

        AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS heart_rate,
        AVG(CASE WHEN ce.itemid IN (220052, 220181) THEN ce.valuenum END) AS mean_arterial_pressure,
        AVG(CASE WHEN ce.itemid = 220210 THEN ce.valuenum END) AS respiratory_rate,
        AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) AS spo2,

        -- 223761 is Fahrenheit and 223762 is Celsius. Convert Fahrenheit to Celsius.
        AVG(
            CASE
                WHEN ce.itemid = 223761 THEN (ce.valuenum - 32.0) * 5.0 / 9.0
                WHEN ce.itemid = 223762 THEN ce.valuenum
            END
        ) AS temperature

    FROM time_windows tw
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ce
        ON tw.stay_id = ce.stay_id
       AND ce.charttime >= DATETIME(tw.window_start)
       AND ce.charttime < DATETIME(tw.window_end)
       AND ce.itemid IN (
            220045,
            220052,
            220181,
            220210,
            220277,
            223761,
            223762
       )
       AND ce.valuenum IS NOT NULL

    GROUP BY tw.stay_id, tw.timestep
),

labs AS (
    SELECT
        tw.subject_id,
        tw.hadm_id,
        tw.stay_id,
        tw.timestep,

        AVG(CASE WHEN le.itemid = 50813 THEN le.valuenum END) AS lactate,
        AVG(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine,
        AVG(CASE WHEN le.itemid IN (51300, 51301) THEN le.valuenum END) AS wbc,
        AVG(CASE WHEN le.itemid = 51265 THEN le.valuenum END) AS platelets,
        AVG(CASE WHEN le.itemid = 50885 THEN le.valuenum END) AS bilirubin

    FROM time_windows tw
    LEFT JOIN `physionet-data.mimiciv_3_1_hosp.labevents` le
        ON tw.subject_id = le.subject_id
       AND tw.hadm_id = le.hadm_id
       AND le.charttime >= DATETIME(tw.window_start)
       AND le.charttime < DATETIME(tw.window_end)
       AND le.itemid IN (
            50813,
            50912,
            51300,
            51301,
            51265,
            50885
       )
       AND le.valuenum IS NOT NULL

    GROUP BY tw.subject_id, tw.hadm_id, tw.stay_id, tw.timestep
),

fluids AS (
    SELECT
        tw.stay_id,
        tw.timestep,
        SUM(ie.amount) AS fluid_amount

    FROM time_windows tw
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.inputevents` ie
        ON tw.stay_id = ie.stay_id
       AND ie.starttime >= DATETIME(tw.window_start)
       AND ie.starttime < DATETIME(tw.window_end)
       AND ie.amount IS NOT NULL
       AND LOWER(ie.ordercategoryname) LIKE '%fluid%'

    GROUP BY tw.stay_id, tw.timestep
),

vasopressors AS (
    SELECT
        tw.stay_id,
        tw.timestep,
        AVG(ie.rate) AS vasopressor_rate

    FROM time_windows tw

    LEFT JOIN `physionet-data.mimiciv_3_1_icu.inputevents` ie
        ON tw.stay_id = ie.stay_id
       AND ie.starttime >= DATETIME(tw.window_start)
       AND ie.starttime < DATETIME(tw.window_end)
       AND ie.rate IS NOT NULL

    LEFT JOIN `physionet-data.mimiciv_3_1_icu.d_items` di
        ON ie.itemid = di.itemid

    WHERE LOWER(di.label) LIKE '%norepinephrine%'
       OR LOWER(di.label) LIKE '%epinephrine%'
       OR LOWER(di.label) LIKE '%vasopressin%'
       OR LOWER(di.label) LIKE '%phenylephrine%'
       OR LOWER(di.label) LIKE '%dopamine%'

    GROUP BY tw.stay_id, tw.timestep
)

SELECT
    tw.subject_id,
    tw.hadm_id,
    tw.stay_id,
    tw.timestep,
    tw.window_start,
    tw.window_end,

    p.gender,
    p.anchor_age,

    a.hospital_expire_flag,

    v.heart_rate,
    v.mean_arterial_pressure,
    v.respiratory_rate,
    v.spo2,
    v.temperature,

    l.lactate,
    l.creatinine,
    l.wbc,
    l.platelets,
    l.bilirubin,

    COALESCE(f.fluid_amount, 0) AS fluid_amount,
    COALESCE(vaso.vasopressor_rate, 0) AS vasopressor_rate

FROM time_windows tw

JOIN infection_flags inf
    ON tw.hadm_id = inf.hadm_id

JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON tw.subject_id = p.subject_id

JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON tw.hadm_id = a.hadm_id

LEFT JOIN vitals v
    ON tw.stay_id = v.stay_id
   AND tw.timestep = v.timestep

LEFT JOIN labs l
    ON tw.subject_id = l.subject_id
   AND tw.hadm_id = l.hadm_id
   AND tw.stay_id = l.stay_id
   AND tw.timestep = l.timestep

LEFT JOIN fluids f
    ON tw.stay_id = f.stay_id
   AND tw.timestep = f.timestep

LEFT JOIN vasopressors vaso
    ON tw.stay_id = vaso.stay_id
   AND tw.timestep = vaso.timestep

WHERE tw.timestep <= {int(max_timestep)}

ORDER BY tw.stay_id, tw.timestep

LIMIT {int(row_limit)}
"""
    return query


# ---------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------

def zero_aware_quantile_bins(series: pd.Series, n_bins: int = 5) -> pd.Series:
    """
    Create treatment intensity bins where zero treatment is always bin 0.

    Positive values are split into bins 1 to n_bins - 1 using quantiles.
    This avoids the major problem where many zero-treatment rows are spread
    across bins because of rank-based qcut.
    """
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    bins = pd.Series(np.zeros(len(values), dtype=int), index=values.index)

    positive_mask = values > 0
    positive_values = values[positive_mask]

    if len(positive_values) == 0:
        return bins.astype(int)

    try:
        ranked = positive_values.rank(method="first")
        positive_bins = pd.qcut(
            ranked,
            q=n_bins - 1,
            labels=False,
            duplicates="drop",
        )
        positive_bins = positive_bins.astype(int) + 1
        bins.loc[positive_mask] = positive_bins
    except ValueError:
        bins.loc[positive_mask] = 1

    return bins.clip(lower=0, upper=n_bins - 1).astype(int)


def compute_sofa_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Compute the same approximate SOFA-like severity proxy used conceptually
    by SepsisTrajectoryEnv. This is saved for inspection only; the environment
    also computes sofa_proxy internally.
    """
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    platelets = pd.to_numeric(df["platelets"], errors="coerce")
    bilirubin = pd.to_numeric(df["bilirubin"], errors="coerce")
    map_value = pd.to_numeric(df["mean_arterial_pressure"], errors="coerce")
    creatinine = pd.to_numeric(df["creatinine"], errors="coerce")
    spo2 = pd.to_numeric(df["spo2"], errors="coerce")

    score += np.select(
        [
            platelets < 50,
            platelets < 100,
            platelets < 150,
            platelets < 200,
        ],
        [4, 3, 2, 1],
        default=0,
    )

    score += np.select(
        [
            bilirubin >= 12.0,
            bilirubin >= 6.0,
            bilirubin >= 2.0,
            bilirubin >= 1.2,
        ],
        [4, 3, 2, 1],
        default=0,
    )

    score += np.select(
        [
            map_value < 50,
            map_value < 65,
            map_value < 70,
        ],
        [3, 2, 1],
        default=0,
    )

    score += np.select(
        [
            creatinine >= 5.0,
            creatinine >= 3.5,
            creatinine >= 2.0,
            creatinine >= 1.2,
        ],
        [4, 3, 2, 1],
        default=0,
    )

    score += np.select(
        [
            spo2 < 85,
            spo2 < 90,
            spo2 < 94,
        ],
        [3, 2, 1],
        default=0,
    )

    return score.astype(float)


def postprocess_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean extracted trajectories and create 25-action treatment labels.
    """
    df = df.copy()
    df = df.sort_values(["stay_id", "timestep"]).reset_index(drop=True)

    # Encode gender for convenience. SepsisTrajectoryEnv also creates this column.
    df["gender_encoded"] = df["gender"].map({"M": 1.0, "F": 0.0}).fillna(0.5)

    numeric_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "timestep",
        "anchor_age",
        "hospital_expire_flag",
        "heart_rate",
        "mean_arterial_pressure",
        "respiratory_rate",
        "spo2",
        "temperature",
        "lactate",
        "creatinine",
        "wbc",
        "platelets",
        "bilirubin",
        "fluid_amount",
        "vasopressor_rate",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    clinical_cols = [
        "heart_rate",
        "mean_arterial_pressure",
        "respiratory_rate",
        "spo2",
        "temperature",
        "lactate",
        "creatinine",
        "wbc",
        "platelets",
        "bilirubin",
    ]

    # Forward-fill and backward-fill within stay, then global median-fill.
    for col in clinical_cols:
        df[col] = df.groupby("stay_id")[col].ffill()
        df[col] = df.groupby("stay_id")[col].bfill()

        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0.0

        df[col] = df[col].fillna(median_value)

    # Treatment variables
    df["fluid_amount"] = df["fluid_amount"].fillna(0.0).clip(lower=0.0)
    df["vasopressor_rate"] = df["vasopressor_rate"].fillna(0.0).clip(lower=0.0)

    # Outcome and demographics
    df["hospital_expire_flag"] = df["hospital_expire_flag"].fillna(0).astype(int)
    df["anchor_age"] = df["anchor_age"].fillna(df["anchor_age"].median())

    # Zero-aware action bins
    df["fluid_bin"] = zero_aware_quantile_bins(df["fluid_amount"], n_bins=5)
    df["vasopressor_bin"] = zero_aware_quantile_bins(df["vasopressor_rate"], n_bins=5)

    df["action"] = (df["fluid_bin"] * 5 + df["vasopressor_bin"]).astype(int)
    df["action"] = df["action"].clip(lower=0, upper=24)

    # Optional saved severity proxy for inspection and analysis.
    df["sofa_proxy"] = compute_sofa_proxy(df)

    # Keep only stays with at least two windows because RL transitions require s_t and s_t+1.
    counts = df.groupby("stay_id").size()
    valid_stays = counts[counts >= 2].index
    df = df[df["stay_id"].isin(valid_stays)].copy()
    df = df.sort_values(["stay_id", "timestep"]).reset_index(drop=True)

    return df


def write_metadata(df: pd.DataFrame, output_path: Path, max_timestep: int, row_limit: int) -> None:
    metadata = {
        "output_path": str(output_path),
        "num_rows": int(len(df)),
        "num_unique_stays": int(df["stay_id"].nunique()),
        "min_timestep": int(df["timestep"].min()) if len(df) else None,
        "max_timestep_in_data": int(df["timestep"].max()) if len(df) else None,
        "query_max_timestep": int(max_timestep),
        "row_limit": int(row_limit),
        "action_space": 25,
        "action_formula": "action = fluid_bin * 5 + vasopressor_bin",
        "fluid_binning": "zero treatment -> bin 0; positive treatment -> quantile bins 1-4",
        "vasopressor_binning": "zero treatment -> bin 0; positive treatment -> quantile bins 1-4",
        "columns": list(df.columns),
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved metadata to {metadata_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-IV v3.1 temporal sepsis trajectories for RL."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path. Default saves to ../data/sepsis_trajectories.csv relative to this script.",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=20,
        help="Maximum 4-hour timestep to keep from each ICU stay.",
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        default=200000,
        help="Maximum rows returned by BigQuery.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print query and exit without running BigQuery.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = build_query(max_timestep=args.max_timestep, row_limit=args.row_limit)

    if args.dry_run:
        print(query)
        return

    print("Running MIMIC-IV temporal trajectory extraction...")
    print(f"Output path: {output_path}")

    client = bigquery.Client()
    df = client.query(query).to_dataframe()

    print("\nPreview before processing:")
    print(df.head())

    print("\nRows extracted:", len(df))
    print("Unique ICU stays:", df["stay_id"].nunique())

    print("\nMissing values before processing:")
    print(df.isna().sum())

    df = postprocess_trajectories(df)

    print("\nMissing values after processing:")
    print(df.isna().sum())

    print("\nRows after filtering valid multi-step stays:", len(df))
    print("Unique ICU stays after filtering:", df["stay_id"].nunique())

    print("\nAction distribution:")
    print(df["action"].value_counts().sort_index())

    print("\n5x5 action heatmap counts:")
    heatmap = pd.crosstab(df["fluid_bin"], df["vasopressor_bin"])
    heatmap = heatmap.reindex(index=range(5), columns=range(5), fill_value=0)
    print(heatmap)

    df.to_csv(output_path, index=False)
    print(f"\nSaved temporal trajectory dataset to {output_path}")

    write_metadata(
        df=df,
        output_path=output_path,
        max_timestep=args.max_timestep,
        row_limit=args.row_limit,
    )


if __name__ == "__main__":
    main()
