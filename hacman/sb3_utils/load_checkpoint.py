from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

import os

def load_checkpoint(path, model: OffPolicyAlgorithm):
    assert path is not None
    
    if path == "latest":
        #TODO: find the path to the latest checkpoint
        pass
    
    dir = os.path.dirname(path)
    ckpt_name = path.split("/")[-1].rstrip(".zip")
    step_num = ckpt_name.split("_")[-2]
    
    replay_name = f"rl_model_replay_buffer_{step_num}_steps.pkl"
    replay_path = os.path.join(dir, replay_name)
    
    loaded_model = model.__class__.load(path, env=model.get_env())
    loaded_model.load_replay_buffer(replay_path)
    
    return loaded_model
        