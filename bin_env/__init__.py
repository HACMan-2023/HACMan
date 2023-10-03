from bin_env.base_env import BaseEnv
from bin_env.poke_env import PokeEnv
from robosuite.environments.base import register_env

register_env(BaseEnv)
register_env(PokeEnv)
