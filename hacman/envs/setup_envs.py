import numpy as np
import functools

from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from hacman.envs.location_policy_wrappers import DummyVecEnvwithLocationPolicy, SubprocVecEnvwithLocationPolicy
from hacman.envs.setup_location_policy import setup_location_policy, add_location_policy_config


def add_env_config(parser):
    parser.add_argument("--env", default='simple_env', type=str, choices={'simple_env', 'hacman_bin_env'})
    parser.add_argument("--train_n_envs", default=20, type=int, help="Number of training envs in parallel")
    parser.add_argument("--eval_n_envs", default=10, type=int, help="Number of eval envs in parallel")
    parser.add_argument("--record_from_cam", default=None, type=str, help="Record video from camera.")
    parser.add_argument("--max_episode_steps", default=10, type=int, help="Max episode steps")

    # Location policy
    add_location_policy_config(parser)
    
    '''
    HACMan Bin Env Specific Configs
    '''
    # Object mesh
    parser.add_argument("--object_dataset", default='housekeep_all', type=str, choices={'cube', 'housekeep', 'housekeep_all'})
    parser.add_argument("--object_types", default=None, nargs='*', type=str, help="Load from specific categories")
    parser.add_argument("--object_name", default=None, type=str, help="Load from a specific object name")
    parser.add_argument("--object_split_train", default='train_thick', type=str, help="Load from a specific split")
    parser.add_argument("--object_split_eval", default='train_thick', type=str, help="Load from a specific split")
    parser.add_argument("--object_scale_range", default=[0.8, 1.2], nargs=2, type=float, help="Scale range for object size randomization")
    parser.add_argument("--object_size_limit", default=[0.05, 0.05], nargs=2, type=float, help="Max/min object size")

    parser.add_argument("--object_voxel_down_sample", default=0.005, type=float, help="Downsample object point cloud voxel size")
    parser.add_argument("--pcd_mode", default='reduced1', type=str, help="Point cloud sampling mode")
    parser.add_argument("--pcd_noise", default=None, nargs=4, type=float, help="Pcd noise")

    # Task
    parser.add_argument("--object_init_pose", default="any", type=str, help="Init pose sampling mode")
    parser.add_argument("--goal_mode", default='any_var_size', type=str, help="Goal sampling mode")
    parser.add_argument("--goal_mode_eval", default=None, type=str, help="Goal sampling mode")
    parser.add_argument("--success_threshold", default=0.03, type=float, help="Success threshold")
    
    # Bin
    parser.add_argument("--table_full_size", default=(0.45, 0.54, 0.107), nargs=3, type=float, help="Bin size")
    parser.add_argument("--table_size_rand", default=(0., 0., 0.), nargs=3, type=float, help="Bin size")
    parser.add_argument("--gripper_types", default='default', type=str, help="Gripper type")
    parser.add_argument("--friction_config", default='default', type=str, help="friction settings")

    # Actions
    parser.add_argument("--action_repeat", default=3, type=int, help="Number of action repeats")
    parser.add_argument("--location_noise", default=0., type=float, help="Location noise")
    parser.add_argument("--reward_scale", default=1., type=float, help="Reward scale")
    parser.add_argument("--fixed_ep_len", action="store_true", help="Fixed episode length")
    parser.add_argument("--reward_gripper_distance", default=None, type=float, help="Whether to use distance in the reward calculation")

    return

def setup_envs(config, eval_env_only=False): 
    train_env, eval_env = None, None
    # ------------- Train/eval wrapper setup ------------- 
    if config["action_mode"] == "per_point_action":
        location_model_train, location_model_eval = setup_location_policy(config)
        if config['debug']:
            train_wrapper = functools.partial(DummyVecEnvwithLocationPolicy, location_model=location_model_train)
            eval_wrapper = functools.partial(DummyVecEnvwithLocationPolicy, location_model=location_model_eval)
        else:
            train_wrapper = functools.partial(SubprocVecEnvwithLocationPolicy, location_model=location_model_train)
            eval_wrapper = functools.partial(SubprocVecEnvwithLocationPolicy, location_model=location_model_eval)
    else:
        if config['debug']:
            train_wrapper, eval_wrapper = DummyVecEnv, DummyVecEnv
        else:
            train_wrapper, eval_wrapper = SubprocVecEnv, SubprocVecEnv
    
    # ------------- Env Setup ------------- 
    env_configs = dict(env=config['env'],
                        reward_scale=config['reward_scale'],
                        reward_gripper_distance=config['reward_gripper_distance'],
                        goal_mode=config['goal_mode'],
                        pcd_mode=config['pcd_mode'],
                        action_mode=config['action_mode'],
                        action_repeat=config['action_repeat'],
                        object_dataset=config['object_dataset'],
                        object_types=config['object_types'],
                        object_name=config['object_name'],
                        object_scale_range=config['object_scale_range'],
                        object_init_pose=config['object_init_pose'],
                        object_split=config['object_split_train'],
                        object_size_limit=config['object_size_limit'],
                        fixed_ep_len=config['fixed_ep_len'],
                        table_full_size=config['table_full_size'],
                        table_size_rand=config['table_size_rand'],
                        location_noise=config['location_noise'],
                        pcd_noise=config['pcd_noise'],
                        object_voxel_down_sample=config['object_voxel_down_sample'],
                        friction_config=config['friction_config'],
                        success_threshold=config['success_threshold'],
                        gripper_types=config['gripper_types'],
                        max_episode_steps=config['max_episode_steps'],
                        )

    def make_env(config):
        if config['env'] == 'hacman_bin_env':
            from hacman.envs.sim_envs.hacman_bin_env import HACManBinEnv
            env = HACManBinEnv(**config)
        elif config['env'] == 'simple_env':
            from hacman.envs.sim_envs.simple_env import SimpleEnv
            env = SimpleEnv(**config)
        else:
            raise NotImplementedError
        env = TimeLimit(env, max_episode_steps=config['max_episode_steps'])
        return env

    # Train Env
    if not eval_env_only:
        make_train_env = functools.partial(make_env, env_configs)
        train_env = make_vec_env(make_train_env, n_envs=config['train_n_envs'], seed=config["seed"], 
                        vec_env_cls=train_wrapper)
    
    # Eval Env
    eval_env_config = env_configs.copy()
    if config['goal_mode_eval'] is not None:
        eval_env_config.update(goal_mode=config['goal_mode_eval'])
    eval_env_config.update(record_from_cam=config['record_from_cam'])
    eval_env_config.update(object_split=config['object_split_eval'])
    make_eval_env = functools.partial(make_env, eval_env_config)
    eval_env = make_vec_env(make_eval_env, n_envs=config['eval_n_envs'], seed=config["seed"], 
                            vec_env_cls=eval_wrapper)
    
    if eval_env_only:
        return eval_env
    
    return train_env, eval_env
