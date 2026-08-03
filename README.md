# Temporal Reinforcement Learning for Sequential Sepsis Treatment Optimization Using MIMIC-IV ICU Trajectories

> Ongoing research project. The manuscript is in progress and has not yet been peer-reviewed or published. No MIMIC-IV patient-level data or restricted clinical data are included in this repository.

## Overview

This project develops a temporal reinforcement learning framework for sequential sepsis treatment optimization using MIMIC-IV ICU trajectories. ICU stays are represented as patient episodes, and clinical progression is modeled using 4-hour time windows. The treatment action space combines intravenous fluid and vasopressor intensity into 25 discrete actions.

The project compares:

- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)
- Offline Conservative Q-Learning (CQL)
- Observed clinician treatment behavior

## Key Features

- Temporal ICU trajectory construction using 4-hour clinical windows
- 15-dimensional clinical state representation
- 25-action fluid–vasopressor treatment space
- PPO and A2C actor-critic training
- Offline CQL baseline
- Off-policy evaluation diagnostics
- Treatment-action distribution analysis
- Severity-aware policy behavior analysis
- PPO ablation study

## Data

This project uses MIMIC-IV, which requires credentialed access through PhysioNet. No MIMIC-IV data, extracted patient-level data, or restricted files are shared in this repository.

Users must obtain their own access to MIMIC-IV and follow all PhysioNet data use requirements.

## Project Status

This project is currently in progress. The codebase is being organized for reproducibility, and the manuscript is still under development.

## Repository Structure

```text
scripts/
  new/
    sepsis_temporal_env.py
    extract_trajectories.py
    train_rl_agent.py
    train_a2c_agent.py
    prepare_offline_dataset_env_aligned.py
    train_cql_agent.py
    evaluate_ppo_a2c_agents.py
    evaluate_cql_agent.py
    analyze_actions.py
    compare_rl_baselines.py
    train_ppo_ablation.py
    evaluate_ppo_ablation.py

figures/
  selected result figures only

results/
  summary CSV files only, no patient-level data
