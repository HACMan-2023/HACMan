import numpy as np
import torch
from stable_baselines3.common.preprocessing import preprocess_obs as pobs


class LocationPolicy(object):
    def __init__(self) -> None:
        pass
        
    def load_model(self, model):
        pass
    
    def get_action(self, obs):
        pass


class LocationPolicyWithArgmaxQ(LocationPolicy):
    def __init__(self, model=None, temperature=1., deterministic=False, vis_only=False, egreedy=0.) -> None:
        self.model = model
        self.temperature = temperature
        self.egreedy = egreedy
        self.deterministic = deterministic
        self.vis_only = vis_only # only use the scores for visualization
        return
    
    def load_model(self, model):
        self.model = model
        return
    
    def get_action(self, obs):  
        assert self.model, "LocationPolicyWithArgmaxQ: needs to load a model."      
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        obs_tensor = {}
        for key in self.model.actor.observation_space.spaces.keys():
            try:
                if key != 'action_location_score':
                    obs_tensor[key] = torch.tensor(obs[key]).to(device).unsqueeze(dim=0)
            except:
                print(f'obs_tensor[key] = torch.tensor(obs[key]).to(device).unsqueeze(dim=0)')
                print(f'key:{key}, value:{obs[key]}')
        
        # Need to use eval. Otherwise this will affect batch norm.
        was_training = self.model.policy.training
        self.model.policy.set_training_mode(False)
        with torch.no_grad():
            preprocessed_obs = pobs(obs_tensor, self.model.actor.observation_space, normalize_images=self.model.actor.normalize_images)
            
            if self.model.actor.features_extractor.__class__.__name__ == 'PointCloudGlobalExtractor':
                # Get action
                actions = self.model.actor(preprocessed_obs)
                # Get per point feature
                features = self.model.critic.features_extractor(preprocessed_obs, per_point_output=True)
                # Concatenate with per point critic feature and get per point value
                actions = torch.cat([actions]*features.shape[0], dim=0)
                out = self.model.critic.q_networks[0](torch.cat([features, actions], dim=1))
            else:
                # Get per point action
                features = self.model.actor.features_extractor(preprocessed_obs, per_point_output=True)
                actions = self.model.actor.mu(features)
                # Get per point feature
                features = self.model.critic.features_extractor(preprocessed_obs, per_point_output=True)
                # Concatenate with per point critic feature and get per point value
                out = self.model.critic.q_networks[0](torch.cat([features, actions], dim=1))
        self.model.policy.set_training_mode(was_training)
        
        action_score = out[: obs['object_pcd_points'].shape[0]].reshape(-1)
        action_score = torch.nn.Softmax(dim=0)(action_score/self.temperature).cpu().detach().numpy()
        
        if self.vis_only:
            poke_idx = np.random.randint(len(obs['object_pcd_points']))
        elif self.deterministic:
            poke_idx = np.argmax(action_score)
        else:
            # Uniformly random before learning_starts 
            if self.model.num_timesteps < self.model.learning_starts or np.random.rand() < self.egreedy:
                poke_idx = np.random.choice(np.arange(len(action_score)))
            else:
                poke_idx = np.random.choice(np.arange(len(action_score)), p=action_score)
        
        poke_idx = np.expand_dims(poke_idx, axis=0)

        action = {'action_location_score':action_score,
                  'poke_idx': poke_idx}
        return action

class RandomLocation(LocationPolicy):
    def __init__(self) -> None:
        return
    
    def get_action(self, obs):
        # Choose an index from the observation point cloud
        points = obs['object_pcd_points']
        if points.ndim == 2:
            poke_idx = np.random.randint(len(points), size=(1,))
            return {'poke_idx': poke_idx,
                    'action_location_score': np.zeros(len(points))}
        elif points.ndim == 3: # batched
            bs, n_points, _ = points.shape
            poke_idx = np.random.randint(n_points, size=(bs, 1))
            return {'poke_idx': poke_idx,
                    'action_location_score': np.zeros((bs, n_points))}
