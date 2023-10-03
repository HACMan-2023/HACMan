import os

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3.common.callbacks import BaseCallback

class CustomizedCheckpointCallback(BaseCallback):
    """
    Modified based on CheckpointCallback
    
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq_latest: Save latest model and replay buffer every ``save_freq_latest`` call of the callback.
    :param save_freq_checkpoint: Save model checkpoints every ``save_freq_checkpoint`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq_latest: int,
        save_freq_checkpoint: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq_latest = save_freq_latest
        self.save_freq_checkpoint = save_freq_checkpoint
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "", latest=False) -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        if latest:
            return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}latest.{extension}")
        else:
            return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq_latest == 0:
            self._save(latest=True, save_replay_buffer=self.save_replay_buffer)
        if self.n_calls % self.save_freq_checkpoint == 0:
            self._save(latest=False)
        return True
    
    def _save(self, latest=False, save_replay_buffer=False):
        model_path = self._checkpoint_path(extension="zip", latest=latest)
        self.model.save(model_path)
        if self.verbose >= 2:
            print(f"Saving model checkpoint to {model_path}")

        if save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            # If model has a replay buffer, save it too
            replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl", latest=latest)
            self.model.save_replay_buffer(replay_buffer_path)
            if self.verbose > 1:
                print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            # Save the VecNormalize statistics
            vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl", latest=latest)
            self.model.get_vec_normalize_env().save(vec_normalize_path)
            if self.verbose >= 2:
                print(f"Saving model VecNormalize to {vec_normalize_path}")
        return True


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )
        self.pbar = None

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Close progress bar
        self.pbar.close()