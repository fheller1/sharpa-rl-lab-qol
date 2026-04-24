# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import dataclasses
import sys
import shutil

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16384, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--cache", type=str, default=None, help="Cache path.")
parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default=None, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
parser.add_argument("--finetune_dataset_dir", type=str, default=None, help="Dir to finetune dataset.")
parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
# domain randomization overrides
parser.add_argument("--reset_random_quat", action="store_true", default=False, help="Randomize hand orientation each episode reset.")
parser.add_argument("--scale_range", type=float, nargs=3, metavar=("MIN", "MAX", "N"), default=None, help="Object scale range [min max n_steps].")
parser.add_argument("--no_randomize_pd_gains", action="store_true", default=False, help="Disable PD gain randomization.")
parser.add_argument("--no_randomize_friction", action="store_true", default=False, help="Disable friction randomization.")
parser.add_argument("--no_randomize_com", action="store_true", default=False, help="Disable center-of-mass randomization.")
parser.add_argument("--no_randomize_mass", action="store_true", default=False, help="Disable object mass randomization.")
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

def _cfg_to_dict(cfg, prefix=""):
    result = {}
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        for f in dataclasses.fields(cfg):
            key = f"{prefix}{f.name}" if prefix else f.name
            val = getattr(cfg, f.name)
            if dataclasses.is_dataclass(val) and not isinstance(val, type):
                result.update(_cfg_to_dict(val, prefix=f"{key}/"))
            elif isinstance(val, (int, float, bool, str)):
                result[key] = val
            elif isinstance(val, (tuple, list)) and all(isinstance(x, (int, float, bool, str)) for x in val):
                result[key] = list(val)
    return result


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree('outputs/')
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["algorithm"]["max_agent_steps"] = args_cli.max_agent_steps if args_cli.max_agent_steps is not None else agent_cfg["algorithm"]["max_agent_steps"]
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs if args_cli.num_envs is not None else agent_cfg["algorithm"]["num_actors"]
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path
    agent_cfg["algorithm"]['minibatch_size'] = min([args_cli.num_envs * 8, 32768])
    if agent_cfg["algo"] == "ProprioAdapt":
        env_cfg.gravity_curriculum = False
    # domain randomization overrides
    if args_cli.reset_random_quat:
        env_cfg.reset_random_quat = True
    if args_cli.scale_range is not None:
        env_cfg.scale_range = [args_cli.scale_range[0], args_cli.scale_range[1], int(args_cli.scale_range[2])]
        env_cfg.events.rand_params(env_cfg.scale_range)
    if args_cli.no_randomize_pd_gains:
        env_cfg.randomize_pd_gains = False
    if args_cli.no_randomize_friction:
        env_cfg.randomize_friction = False
    if args_cli.no_randomize_com:
        env_cfg.randomize_com = False
    if args_cli.no_randomize_mass:
        env_cfg.randomize_mass = False
    config = ConfigWrapper(agent_cfg, env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    if agent_cfg["algo"] in ["ProprioAdapt"]:
        load_path_split = agent_cfg["load_path"].split("/")
        if agent_cfg["algorithm"]["experiment_name"] in load_path_split and "stage1_nn" in load_path_split:
            log_dir = os.path.join(*(load_path_split[-5:-2]))
    print(f"Exact experiment name requested from command line: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config)

    spec = gym.spec(args_cli.task)
    env_cfg_file = spec.kwargs.get("env_cfg_entry_point", None).split(":")[0].replace(".", "/") + ".py"
    agent_cfg_file = spec.kwargs.get("agent_cfg_entry_point", None).replace(".", "/").replace(":", "/").replace("/yaml", ".yaml")
    shutil.copy(env_cfg_file, os.path.join(log_dir, f"env_cfg_{agent_cfg['algo']}.py"))
    shutil.copy(agent_cfg_file, os.path.join(log_dir, f"agent_cfg_{agent_cfg['algo']}.yaml"))

    # load the checkpoint
    if args_cli.resume or agent_cfg["algo"] in ["ProprioAdapt"]:
        resume_path = agent_cfg["load_path"]
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        agent.restore_train(resume_path)

    if args_cli.wandb:
        import wandb
        wandb.init(project=agent_cfg["algorithm"]["experiment_name"], dir=log_dir, config={
            **agent_cfg["algorithm"],
            "num_envs": args_cli.num_envs,
            "seed": agent_cfg["seed"],
            **{f"env/{k}": v for k, v in _cfg_to_dict(env_cfg).items()},
        })

    # run training
    agent.train()
    
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
