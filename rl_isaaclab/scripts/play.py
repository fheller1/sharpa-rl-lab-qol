# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys
import shutil

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--cache", type=str, default=None, help="Cache path.")
parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default=None, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Save rollout video.")
parser.add_argument("--video_length", type=int, default=400, help="Number of steps per video clip.")
parser.add_argument("--camera_eye", type=float, nargs=3, default=[0.5, 0.5, 0.7], metavar=("X", "Y", "Z"), help="Viewport camera position.")
parser.add_argument("--camera_target", type=float, nargs=3, default=[0.0, 0.0, 0.6], metavar=("X", "Y", "Z"), help="Viewport camera look-at target.")
# domain randomization overrides
parser.add_argument("--reset_random_quat", action="store_true", default=False, help="Randomize hand orientation each episode reset.")
parser.add_argument("--scale_range", type=float, nargs=3, metavar=("MIN", "MAX", "N"), default=None, help="Object scale range [min max n_steps].")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg

import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree('outputs/')
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["algorithm"]["max_agent_steps"] = args_cli.max_agent_steps if args_cli.max_agent_steps is not None else 2000
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    env_cfg.reset_random_quat = args_cli.reset_random_quat
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    env_cfg.gravity_curriculum = False
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path
    if args_cli.scale_range is not None:
        env_cfg.scale_range = [args_cli.scale_range[0], args_cli.scale_range[1], int(args_cli.scale_range[2])]
        env_cfg.events.rand_params(env_cfg.scale_range)
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        from isaaclab.sim import SimulationContext
        _eye = args_cli.camera_eye
        _target = args_cli.camera_target
        def _set_cam():
            sim = SimulationContext.instance()
            if sim is not None:
                sim.set_camera_view(eye=_eye, target=_target)
        _orig_reset = env.reset
        def _reset_with_cam(*a, **kw):
            out = _orig_reset(*a, **kw)
            _set_cam()
            return out
        env.reset = _reset_with_cam
        _set_cam()
        video_kwargs = {"video_folder": os.path.join(log_dir, "videos"), "step_trigger": lambda s: s == 0, "video_length": args_cli.video_length, "disable_logger": True}
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)
    
    # load the checkpoint
    resume_path = agent_cfg["load_path"]
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent.restore_test(resume_path)
    agent.test()

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
