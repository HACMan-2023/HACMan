from hacman.algos.location_policy import LocationPolicy, LocationPolicyWithArgmaxQ, RandomLocation
from hacman.envs.location_policy_wrappers import SubprocVecEnvwithLocationPolicy, DummyVecEnvwithLocationPolicy


def add_location_policy_config(parser):
    parser.add_argument("--location_model", default='argmaxQ', type=str, help="location model options")
    parser.add_argument("--location_model_temperature", default=0.1, type=float, help="location model options")
    parser.add_argument("--egreedy", default=0.1, type=float, help="Epsilon greedy for location")    
    return

def setup_location_policy(config):
    # Return the location policy for train_env and eval_env
    if config['location_model'] == 'random':
        location_model_train = RandomLocation()
        location_model_eval = location_model_train
    elif config['location_model'] == 'argmaxQ_vis':
        location_model_train = RandomLocation()
        location_model_eval = LocationPolicyWithArgmaxQ(model=None, vis_only=True) # Load model later
        return location_model_train, location_model_eval
    elif config['location_model'] == 'argmaxQ':
        location_model_train = LocationPolicyWithArgmaxQ(model=None, 
                                                        temperature=config['location_model_temperature'],
                                                        egreedy=config['egreedy'])
        location_model_eval = LocationPolicyWithArgmaxQ(model=None, 
                                                        temperature=config['location_model_temperature'],
                                                        deterministic=True)
    else:
        raise NotImplementedError
    
    return location_model_train, location_model_eval