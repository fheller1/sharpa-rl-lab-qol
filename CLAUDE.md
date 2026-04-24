# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning sim-to-real transfer for training dexterous in-hand object rotation on the SharpaWave robot hand. Training uses Isaac Lab (NVIDIA Isaac Sim) with a two-stage pipeline: PPO with privileged information, then ProprioAdapt distillation for proprioceptive-history-only inference on real hardware.

## Environment Setup

```bash
conda activate env_isaaclab
pip install -e .  # from repo root
```

Requires Isaac Lab 2.2.0 or 2.3.0, Ubuntu 22.04, NVIDIA GPU, 32GB+ RAM.

## Key Commands

```bash
# Stage 0: Generate grasp cache (required before training)
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless

# Stage 1: Train with PPO (privileged info)
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless

# Visualize a trained policy
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path ${pth}

# Stage 2: Distillation (ProprioAdapt) — loads Stage 1 checkpoint
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path ${pth}

# Deploy to real robot (hand_side: 0=left, 1=right)
python rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --hand_side 0 --load_path ${pth}
```

Common flags: `--num_envs`, `--seed`, `--cache`, `--max_agent_steps`, `--resume`, `--device`.

## Architecture

### Training Pipeline

```
gen_grasp.py → train.py (PPO) → [play.py] → train.py (ProprioAdapt) → deploy.py
```

The grasp cache (`cache/sharpa_grasp_linspace_*.npy`) stores valid pre-grasp states across object scales and is required at the start of each training episode.

### Module Structure

- **`rl_isaaclab/scripts/`** — Entry points: `train.py`, `play.py`, `deploy.py`, `gen_grasp.py`
- **`rl_isaaclab/tasks/inhand_rotate/`** — Three Isaac Lab environments (training, grasp gen, deploy), each with a matching `*_cfg.py` and `*_env.py`
- **`rl_isaaclab/algo/`** — PPO (`ppo/ppo.py`) and ProprioAdapt (`padapt/padapt.py`); neural networks in `models/models.py`
- **`rl_isaaclab/wrapper/`** — `GymStyleEnvWrapper` (training), `ConfigWrapper` (merges env+agent config), `vec_env.py`
- **`rl_isaaclab/utils/`** — DOF index mapping helpers, keyboard listener, randomization events

### Registered Gym Environments

| Task ID | Class | Purpose |
|---|---|---|
| `Isaac-Inhand-Rotate-Sharpa-Wave-v0` | `SharpaWaveInhandRotateEnv` | Training |
| `Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0` | `SharpaWaveInhandRotateGraspEnv` | Grasp cache generation |
| `Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0` | `SharpaWaveInhandRotateDeployEnv` | Real robot deployment |

### Environment Specs

- **Action space**: 22 DOF joint torques
- **Observation space**: 192-dim (proprioception + tactile + object state)
- **Privileged info**: 8-dim (friction, mass, COM — Stage 1 only)
- **Proprioception history**: 30-frame buffer (used by ProprioAdapt)
- **Control loop**: 20ms (sim dt=1/240s, decimation=12)
- **Episode length**: 20 seconds

### Neural Network (`algo/models/models.py`)

`ActorCritic` contains:
- **Actor/Critic**: shared MLP trunk with ELU activations
- **Priv info MLP**: processes 8-dim privileged info during Stage 1
- **`ProprioAdaptTConv`**: temporal conv encoder over 30-step proprioception history; added and trained during Stage 2 (base policy frozen)

### Config Pattern

`ConfigWrapper` merges `env_cfg` + agent YAML config (from `agents/ppo_cfg.yaml`). Accessed as `full_config.train["device"]`, `full_config.train["network"]`, etc.

### DOF Mapping

The simulator and real hardware use different joint orderings. Always use `dof_isaaclab2sharpa` / `dof_sharpa2isaaclab` from `rl_isaaclab/utils/misc.py` when exchanging joint commands/observations between sim and hardware.

### Outputs

- Model checkpoints: `logs/{experiment_name}/{timestamp}/*.pth`
- TensorBoard logs: same directory
- Grasp cache: `cache/sharpa_grasp_linspace_{min}-{max}-{n_scales}.npy`

### Real Robot Deployment Notes

- Keyboard control during deployment: `e` = start, `w` = freeze, `q` = home
- Two tactile modes: HostComputer (recommended, higher frequency) or OnBoard (legacy)
- Real robot SDK lives at `rl_isaaclab/utils/python/sharpa/`
- Tactile sensor maps loaded from `assets/tactile_ha4_map/`
