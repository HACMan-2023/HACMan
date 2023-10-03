# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from hacman.utils.launch_utils import use_freer_gpu
use_freer_gpu()
# print(f"Using CUDA Device {os.environ['CUDA_VISIBLE_DEVICES']}")

from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import wandb
from hacman.envs.wandb_wrappers import WandbPointCloudRecorder
from hacman.sb3_utils.custom_callbacks import ProgressBarCallback, CustomizedCheckpointCallback
from hacman.sb3_utils.evaluation import evaluate_policy
from hacman.algos.setup_model import setup_model, add_model_config
from hacman.envs.setup_envs import setup_envs, add_env_config

import argparse
import torch_geometric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Exp and logging config
    parser.add_argument("--ExpID", default=9999, type=int, help="Exp ID")
    parser.add_argument("--name", default="tmp", type=str, help="Exp Name")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--eval", default=None, type=int, help="Eval only. Eval epsiodes to run")
    parser.add_argument("--dirname", default=None, type=str, help="Path to save models")
    parser.add_argument("--debug", action="store_true", help="Use debug config")

    parser.add_argument("--n_eval_episodes", default=20, type=int, help="Number of eval episodes")
    parser.add_argument("--save_freq_latest", default=100, type=int, help="Save per n env steps")
    parser.add_argument("--save_freq_checkpoint", default=1000, type=int, help="Save per n env steps")
    parser.add_argument("--save_replay_buffer", action="store_true", help="Save buffer")
    parser.add_argument("--record_video", action="store_true", help="Record video.")
    parser.add_argument("--upload_video", action="store_true", help="Upload video to wandb.")

    add_model_config(parser)
    add_env_config(parser)
    
    args = parser.parse_args()
    config = vars(args)

    if args.dirname is None or args.debug:
        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    else:
        dirname = args.dirname
    config['dirname'] = dirname
    config["fullname"] = f"Exp{args.ExpID:04d}-{args.name}"
    
    torch_geometric.seed_everything(config["seed"])
    print(config)

    # ------------- Logging ------------- 
    result_folder = os.path.join(config["dirname"], f'{config["fullname"]}-{config["seed"]}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    run_id = wandb.util.generate_id()
    run = wandb.init(# project="gc-reorientation", entity="bwww", 
                     name=config["fullname"], config=config, id=run_id, 
                     dir=result_folder, sync_tensorboard=True)
    print(f'wandb run-id:{run.id}')
    print(f'Result folder: {result_folder}')
    
    # ------------- Task Setup -------------
    if config['eval'] is not None:
        eval_env = setup_envs(config, eval_env_only=True)
        model = setup_model(config, eval_env, result_folder, normalize_param=None)
        
        if hasattr(eval_env, "location_model"):
            eval_env.location_model.load_model(model)

        model.policy.eval()

    else:
        env, eval_env = setup_envs(config)
        model = setup_model(config, env, result_folder, normalize_param=None)
        
        if hasattr(env, "location_model") and hasattr(eval_env, "location_model"):
            env.location_model.load_model(model)
            eval_env.location_model.load_model(model)
            
    # ------------- Additional Setup for Logging Purposes ------------- 
    def global_step_counter():
        return model.num_timesteps
    eval_env = WandbPointCloudRecorder(eval_env, 
        global_step_counter=global_step_counter, 
        save_plotly=True, foldername=result_folder,
        record_video=config['record_video'],
        upload_video=config['upload_video'])
    if config['record_video']:
            eval_env.enable_video_recording()
    
    eval_callback = EvalCallback(
        eval_env, 
        n_eval_episodes=config['n_eval_episodes'], 
        eval_freq=config['eval_freq'],
        best_model_save_path=os.path.join(result_folder, "model"))
    ckpt_callback = CustomizedCheckpointCallback(
        save_freq_latest=config['save_freq_latest'],
        save_freq_checkpoint=config['save_freq_checkpoint'],
        save_path=os.path.join(result_folder, f"model-{run.id}"),
        save_replay_buffer=config['save_replay_buffer'],
        save_vecnormalize=True)  # TODO: confirm?
    progress_bar_callback = ProgressBarCallback()
    callback_list = CallbackList([ckpt_callback, eval_callback, progress_bar_callback])
    
    # ------------- Start Training -------------
    if config['eval'] is None:
        model.learn(
            total_timesteps=1000000,
            log_interval=10,
            callback=callback_list)

    # ------------- Eval -------------
    else:
        model.policy.eval()

        from tqdm import tqdm
        import numpy as np
        import pandas as pd

        pbar = tqdm(total=config['eval'])
        mean_reward, std_reward, succ, verbose_buffer = evaluate_policy(
            model, eval_env, n_eval_episodes=config['eval'], deterministic=True,
            save_path=os.path.join(result_folder, f'obs_list_{0}.pkl'), verbose=True, pbar=pbar)
        
        uncertainty = 1.96 * np.sqrt(succ * (1 - succ) / config['eval'])
        print(f"succ={succ:.3f} +/- {uncertainty:.3f}. mean_reward={mean_reward:.2f} +/- {std_reward}")

        # Export the verbose buffer
        eval_dir = os.path.join(result_folder, 'evals')
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        import pickle
        with open(os.path.join(eval_dir, f'verbose_eval.pkl'), 'wb') as f:
            pickle.dump(verbose_buffer, f)