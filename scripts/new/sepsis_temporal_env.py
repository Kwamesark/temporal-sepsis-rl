import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class SepsisTrajectoryEnv(gym.Env):
    """
    Temporal Gymnasium environment for sepsis treatment optimization.

    One episode = one ICU stay.
    Each step = one 4-hour window.
    State evolves over time using MIMIC-IV trajectory data.
    Action space = 25 treatment bins:
        action = fluid_bin * 5 + vasopressor_bin
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, data_path="../data/sepsis_trajectories.csv"):
        super().__init__()

        self.df = pd.read_csv(data_path)

        required_cols = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "timestep",
            "gender",
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
            "fluid_bin",
            "vasopressor_bin",
            "action",
        ]

        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in trajectory dataset: {missing_cols}")

        self.df = self.df.copy()
        self.df = self.df.sort_values(["stay_id", "timestep"]).reset_index(drop=True)

        self.df["gender_encoded"] = (
            self.df["gender"].map({"M": 1.0, "F": 0.0}).fillna(0.5)
        )

        numeric_cols = [
            "anchor_age",
            "gender_encoded",
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
            "fluid_bin",
            "vasopressor_bin",
            "action",
            "hospital_expire_flag",
        ]

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            median_value = self.df[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            self.df[col] = self.df[col].fillna(median_value)

        self.df["sofa_proxy"] = self.df.apply(self._compute_sofa_proxy_from_row, axis=1)

        self.feature_cols = [
            "anchor_age",
            "gender_encoded",
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
            "sofa_proxy",
            "fluid_amount",
            "vasopressor_rate",
        ]

        # Keep only stays with at least 2 timesteps
        counts = self.df.groupby("stay_id").size()
        valid_stays = counts[counts >= 2].index
        self.df = self.df[self.df["stay_id"].isin(valid_stays)].copy()

        self.trajectories = {
            stay_id: group.sort_values("timestep").reset_index(drop=True)
            for stay_id, group in self.df.groupby("stay_id")
        }

        self.stay_ids = list(self.trajectories.keys())

        if len(self.stay_ids) == 0:
            raise ValueError("No valid multi-step ICU trajectories found.")

        self.current_stay_id = None
        self.current_traj = None
        self.current_timestep = 0

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(25)

    def _decode_action(self, action):
        action = int(action)
        fluid_bin = action // 5
        vasopressor_bin = action % 5
        return fluid_bin, vasopressor_bin

    def _compute_sofa_proxy_from_row(self, row):
        platelets = float(row["platelets"])
        bilirubin = float(row["bilirubin"])
        map_value = float(row["mean_arterial_pressure"])
        creatinine = float(row["creatinine"])
        spo2 = float(row["spo2"])

        score = 0

        if platelets < 50:
            score += 4
        elif platelets < 100:
            score += 3
        elif platelets < 150:
            score += 2
        elif platelets < 200:
            score += 1

        if bilirubin >= 12.0:
            score += 4
        elif bilirubin >= 6.0:
            score += 3
        elif bilirubin >= 2.0:
            score += 2
        elif bilirubin >= 1.2:
            score += 1

        if map_value < 50:
            score += 3
        elif map_value < 65:
            score += 2
        elif map_value < 70:
            score += 1

        if creatinine >= 5.0:
            score += 4
        elif creatinine >= 3.5:
            score += 3
        elif creatinine >= 2.0:
            score += 2
        elif creatinine >= 1.2:
            score += 1

        if spo2 < 85:
            score += 3
        elif spo2 < 90:
            score += 2
        elif spo2 < 94:
            score += 1

        return float(score)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_stay_id = self.np_random.choice(self.stay_ids)
        self.current_traj = self.trajectories[self.current_stay_id]
        self.current_timestep = 0

        obs = self._get_observation()

        row = self.current_traj.iloc[self.current_timestep]

        info = {
            "stay_id": int(self.current_stay_id),
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "timestep": int(row["timestep"]),
            "sofa_proxy": float(row["sofa_proxy"]),
        }

        return obs, info

    def step(self, action):
        action = int(action)

        current_row = self.current_traj.iloc[self.current_timestep]

        next_index = min(self.current_timestep + 1, len(self.current_traj) - 1)
        next_row = self.current_traj.iloc[next_index]

        reward = self._get_reward(action, current_row, next_row)

        self.current_timestep += 1

        terminated = self.current_timestep >= len(self.current_traj) - 1
        truncated = False

        obs = self._get_observation()

        fluid_bin, vasopressor_bin = self._decode_action(action)

        info = {
            "action": action,
            "fluid_bin": fluid_bin,
            "vasopressor_bin": vasopressor_bin,
            "clinician_action": int(current_row["action"]),
            "clinician_fluid_bin": int(current_row["fluid_bin"]),
            "clinician_vasopressor_bin": int(current_row["vasopressor_bin"]),
            "mortality": int(current_row["hospital_expire_flag"]),
            "timestep": int(current_row["timestep"]),
            "lactate": float(current_row["lactate"]),
            "next_lactate": float(next_row["lactate"]),
            "map": float(current_row["mean_arterial_pressure"]),
            "next_map": float(next_row["mean_arterial_pressure"]),
            "sofa_proxy": float(current_row["sofa_proxy"]),
            "next_sofa_proxy": float(next_row["sofa_proxy"]),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        row = self.current_traj.iloc[self.current_timestep]
        obs = row[self.feature_cols].astype(float).values
        return obs.astype(np.float32)

    def _get_reward(self, action, current_row, next_row):
        """
        Temporal reward:
        - reward lactate improvement
        - reward MAP stabilization
        - reward SOFA proxy reduction
        - penalize mismatch from clinician treatment intensity
        - terminal survival/mortality reward
        """

        agent_fluid_bin, agent_vaso_bin = self._decode_action(action)

        clinician_fluid_bin = int(current_row["fluid_bin"])
        clinician_vaso_bin = int(current_row["vasopressor_bin"])

        agent_intensity = agent_fluid_bin + agent_vaso_bin
        clinician_intensity = clinician_fluid_bin + clinician_vaso_bin

        lactate_now = float(current_row["lactate"])
        lactate_next = float(next_row["lactate"])

        map_now = float(current_row["mean_arterial_pressure"])
        map_next = float(next_row["mean_arterial_pressure"])

        sofa_now = float(current_row["sofa_proxy"])
        sofa_next = float(next_row["sofa_proxy"])

        mortality = int(current_row["hospital_expire_flag"])

        reward = 0.0

        # Lactate improvement
        if lactate_next < lactate_now:
            reward += 0.3
        elif lactate_next > lactate_now:
            reward -= 0.3

        # MAP stabilization
        if map_now < 65 and map_next >= 65:
            reward += 0.4
        elif map_now < 65 and map_next < 65:
            reward -= 0.3

        # SOFA proxy improvement
        if sofa_next < sofa_now:
            reward += 0.4
        elif sofa_next > sofa_now:
            reward -= 0.4

        # Severity-aware treatment logic
        if lactate_now >= 4 or sofa_now >= 8 or map_now < 65:
            if agent_intensity == 0:
                reward -= 0.5
            elif agent_vaso_bin >= 1 or agent_fluid_bin >= 1:
                reward += 0.2

        # Avoid overly aggressive treatment in low-risk states
        if lactate_now < 2 and map_now >= 65 and sofa_now <= 2:
            if agent_intensity >= 5:
                reward -= 0.4

        # Behavior cloning-style soft penalty
        action_distance = abs(agent_fluid_bin - clinician_fluid_bin) + abs(
            agent_vaso_bin - clinician_vaso_bin
        )
        reward -= 0.05 * action_distance

        # Terminal outcome reward
        if self.current_timestep >= len(self.current_traj) - 2:
            if mortality == 0:
                reward += 1.0
            else:
                reward -= 1.0

        return float(reward)

    def render(self):
        if self.current_traj is None:
            print("Environment has not been reset.")
            return

        row = self.current_traj.iloc[self.current_timestep]

        print(
            f"Stay ID: {self.current_stay_id}, "
            f"Timestep: {self.current_timestep}, "
            f"Lactate: {row['lactate']:.2f}, "
            f"MAP: {row['mean_arterial_pressure']:.2f}, "
            f"SOFA Proxy: {row['sofa_proxy']:.2f}, "
            f"Clinician Action: {int(row['action'])}"
        )

    def close(self):
        pass