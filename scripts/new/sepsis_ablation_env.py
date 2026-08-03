#!/usr/bin/env python3
"""
sepsis_ablation_env.py

Ablation-compatible sepsis temporal environment.

Place this file in scripts/new/ beside sepsis_temporal_env.py.

Supported ablations:
    full
    no_clinician_penalty
    no_severity_treatment
    no_sofa_proxy
    no_reward_normalization
"""

from typing import List

import numpy as np
from gymnasium import spaces

try:
    from sepsis_temporal_env import SepsisTrajectoryEnv
except ModuleNotFoundError:
    from scripts.new.sepsis_temporal_env import SepsisTrajectoryEnv


VALID_ABLATIONS = {
    "full",
    "no_clinician_penalty",
    "no_severity_treatment",
    "no_sofa_proxy",
    "no_reward_normalization",
}


class SepsisAblationEnv(SepsisTrajectoryEnv):
    """SepsisTrajectoryEnv with reward/state ablation switches."""

    def __init__(self, data_path="../data/sepsis_trajectories.csv", ablation_name="full"):
        if ablation_name not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation_name={ablation_name}. "
                f"Valid options are: {sorted(VALID_ABLATIONS)}"
            )

        self.ablation_name = ablation_name
        super().__init__(data_path=data_path)

        # State-space ablation: remove SOFA-like proxy from observation.
        # Reward still uses sofa_proxy internally to isolate the effect of
        # giving severity information to the policy state.
        if self.ablation_name == "no_sofa_proxy":
            self.feature_cols = [c for c in self.feature_cols if c != "sofa_proxy"]
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.feature_cols),),
                dtype=np.float32,
            )

    def _get_reward(self, action, current_row, next_row):
        """
        Ablation-aware temporal reward.

        Full reward components:
            - lactate improvement
            - MAP stabilization
            - SOFA-like severity reduction
            - severity-aware treatment logic
            - low-risk aggressive-treatment penalty
            - clinician action-distance penalty
            - terminal survival/mortality reward
        """
        agent_fluid_bin, agent_vaso_bin = self._decode_action(action)

        clinician_fluid_bin = int(current_row["fluid_bin"])
        clinician_vaso_bin = int(current_row["vasopressor_bin"])

        agent_intensity = agent_fluid_bin + agent_vaso_bin

        lactate_now = float(current_row["lactate"])
        lactate_next = float(next_row["lactate"])

        map_now = float(current_row["mean_arterial_pressure"])
        map_next = float(next_row["mean_arterial_pressure"])

        sofa_now = float(current_row["sofa_proxy"])
        sofa_next = float(next_row["sofa_proxy"])

        mortality = int(current_row["hospital_expire_flag"])

        reward = 0.0

        # 1. Lactate improvement
        if lactate_next < lactate_now:
            reward += 0.3
        elif lactate_next > lactate_now:
            reward -= 0.3

        # 2. MAP stabilization
        if map_now < 65 and map_next >= 65:
            reward += 0.4
        elif map_now < 65 and map_next < 65:
            reward -= 0.3

        # 3. SOFA proxy improvement
        if sofa_next < sofa_now:
            reward += 0.4
        elif sofa_next > sofa_now:
            reward -= 0.4

        # 4. Severity-aware treatment logic
        if self.ablation_name != "no_severity_treatment":
            if lactate_now >= 4 or sofa_now >= 8 or map_now < 65:
                if agent_intensity == 0:
                    reward -= 0.5
                elif agent_vaso_bin >= 1 or agent_fluid_bin >= 1:
                    reward += 0.2

            # Avoid overly aggressive treatment in low-risk states
            if lactate_now < 2 and map_now >= 65 and sofa_now <= 2:
                if agent_intensity >= 5:
                    reward -= 0.4

        # 5. Behavior cloning-style soft penalty
        if self.ablation_name != "no_clinician_penalty":
            action_distance = abs(agent_fluid_bin - clinician_fluid_bin) + abs(
                agent_vaso_bin - clinician_vaso_bin
            )
            reward -= 0.05 * action_distance

        # 6. Terminal outcome reward
        if self.current_timestep >= len(self.current_traj) - 2:
            if mortality == 0:
                reward += 1.0
            else:
                reward -= 1.0

        return float(reward)


def get_ablation_variants() -> List[str]:
    return [
        "full",
        "no_clinician_penalty",
        "no_severity_treatment",
        "no_sofa_proxy",
        "no_reward_normalization",
    ]
