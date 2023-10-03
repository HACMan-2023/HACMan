import os
import functools

from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3 import TD3

from hacman.algos.hacman_td3 import HACManTD3
from hacman.algos.mix_td3 import MixTD3Policy
from hacman.algos.feature_extractors.feature_extractors import PointCloudExtractor, PointCloudGlobalExtractor, StatesExtractor


def add_model_config(parser):
    parser.add_argument("--load_ckpt", default=None, type=str, help="Ckpt path. Set to \"latest\" to use the latest checkpoint.")
    parser.add_argument("--algo", default='TD3', type=str, help="RL algorithm")
    parser.add_argument("--gradient_steps", default=40, type=int, help="Gradient step per env step")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch Size")
    parser.add_argument("--ent_coef", default='auto', type=str, help="Entropy Coefficient mode")
    parser.add_argument("--clip_grad_norm", default=None, type=float, help="Clip gradient norm for critic")
    parser.add_argument("--clamp_critic_max", default=None, type=float, help="Clamp critic value")
    parser.add_argument("--clamp_critic_min", default=None, type=float, help="Clamp critic value")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--actor_update_interval", default=4, type=int, help="Actor update interval")
    parser.add_argument("--target_update_interval", default=4, type=int, help="Target update interval")
    parser.add_argument("--initial_timesteps", default=10000, type=int, help="Initial env steps before training starts")
    parser.add_argument("--eval_freq", default=100, type=int, help="Eval per n env steps")
    parser.add_argument("--mean_q", action="store_true", help="Use mean Q instead of min Q")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--share_features_extractor", action="store_true", help="share features extractor")
    parser.add_argument("--preprocessing_fn", default="flow", type=str, help="Input processing function")
    parser.add_argument('--net_arch', default=[128, 128, 128], nargs="+", type=int)

    # Baseline options
    parser.add_argument("--action_mode", default='per_point_action', type=str,
                        choices={'per_point_action', "regress_location_and_action", 'regress_action_only'},
                        help="Set action modes for different baselines.")
    parser.add_argument("--feature_mode", default='points', type=str,
                        choices={'points', 'states'})
    return


def setup_model(config, env, result_folder, normalize_param=None):
    # ----------- Feature extractors ------------ 
    if config["feature_mode"] == "states":
        if config["action_mode"] == "regress_location_and_action":
            features_extractor_class = StatesExtractor
        elif config["action_mode"] == "regress_action_only":
            features_extractor_class = functools.partial(StatesExtractor, include_gripper=True)
        else:
            raise ValueError
        
    if config["feature_mode"] == "points":
        if config["action_mode"] == "per_point_action":
            features_extractor_class = functools.partial(PointCloudExtractor, preprocessing_fn=config['preprocessing_fn'])
        elif config["action_mode"] == "regress_location_and_action":
            features_extractor_class = PointCloudGlobalExtractor
        elif config["action_mode"] == "regress_action_only":
            features_extractor_class = functools.partial(PointCloudGlobalExtractor, include_gripper=True)
        else:
            raise ValueError
        
    if config['feature_mode'] == 'points' and normalize_param is not None:
        features_extractor_class =  functools.partial(features_extractor_class, normalize_pos_param=normalize_param)

    # ----------- Model ------------ 
    if config['algo'] == 'TD3':
        policy = functools.partial(TD3Policy,
                        features_extractor_class=features_extractor_class,
                        share_features_extractor=config["share_features_extractor"],
                        net_arch=config['net_arch'])
        td3_kwargs = dict(batch_size=config['batch_size'], gamma=config['gamma'],
                verbose=1, learning_starts=config['initial_timesteps'], learning_rate=config['learning_rate'],
                actor_update_interval=config['actor_update_interval'], 
                target_update_interval=config['target_update_interval'], train_freq=1,
                tensorboard_log=result_folder+'/tb', gradient_steps=config['gradient_steps'], 
                clip_critic_grad_norm=config['clip_grad_norm'], clamp_critic_min=config['clamp_critic_min'],
                clamp_critic_max=config['clamp_critic_max'], seed=config["seed"],
                mean_q=config['mean_q'])
        if config['load_ckpt'] is None:
            model = TD3(policy, env, **td3_kwargs)
        else:
            model = TD3.load(path=config['load_ckpt'], env=env, **td3_kwargs) # Overwrite hyperparameters of the loaded model
            print(f"Loaded policy: {config['load_ckpt']}")
    elif config['algo'] == 'HybridTD3':
        policy = functools.partial(TD3Policy,
                        features_extractor_class=features_extractor_class,
                        share_features_extractor=False,
                        net_arch=config['net_arch'])
        hybrid_td3_kwargs = dict(
            batch_size=config['batch_size'], gamma=config['gamma'],
            verbose=1, learning_starts=config['initial_timesteps'], learning_rate=config['learning_rate'],
            actor_update_interval=config['actor_update_interval'],
            target_update_interval=config['target_update_interval'], train_freq=1,
            tensorboard_log=result_folder+'/tb', gradient_steps=config['gradient_steps'],
            clip_critic_grad_norm=config['clip_grad_norm'], clamp_critic_min=config['clamp_critic_min'],
            clamp_critic_max=config['clamp_critic_max'], seed=config["seed"],
            mean_q=config['mean_q'], temperature=config['location_model_temperature']
        )
        if config['load_ckpt'] is None:
            model = HACManTD3(policy, env, **hybrid_td3_kwargs)
        else:
            model = HACManTD3.load(path=config['load_ckpt'], env=env, **hybrid_td3_kwargs) # Overwrite hyperparameters of the loaded model
            print(f"Loaded policy: {config['load_ckpt']}")
        
    elif config['algo'] == 'TD3MixArch':
        policy = functools.partial(MixTD3Policy,
                        actor_features_extractor_class=PointCloudGlobalExtractor,
                        critic_features_extractor_class=PointCloudExtractor,
                        net_arch=config['net_arch'])
        model = TD3(policy, env, batch_size=config['batch_size'], gamma=config['gamma'],
                verbose=1, learning_starts=config['initial_timesteps'], learning_rate=config['learning_rate'],
                actor_update_interval=config['actor_update_interval'],
                target_update_interval=config['target_update_interval'], train_freq=1,
                tensorboard_log=result_folder+'/tb', gradient_steps=config['gradient_steps'],
                clip_critic_grad_norm=config['clip_grad_norm'], clamp_critic_min=config['clamp_critic_min'],
                clamp_critic_max=config['clamp_critic_max'], seed=config["seed"],
                mean_q=config['mean_q'])

    return model