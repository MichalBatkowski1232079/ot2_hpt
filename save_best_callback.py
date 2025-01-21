from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os


class SaveBestRewardAtEndCallback(BaseCallback):
    def __init__(self, log_dir: str, n_steps: int, verbose=1):
        super(SaveBestRewardAtEndCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.best_model = None
        self.n_steps = n_steps  # Include n_steps in the filename

    def _on_step(self) -> bool:
        # Get the mean reward for the last episodes
        mean_reward = np.mean(self.locals["rewards"])
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.best_model = self.model  # Keep the current best model in memory
            if self.verbose > 0:
                print(f"New best mean reward: {mean_reward:.2f}")
        return True

    def _on_training_end(self) -> None:
        # Save the best model when training ends
        if self.best_model is not None:
            save_path = os.path.join(self.log_dir, f"best_model_nsteps_{self.n_steps}.zip")
            self.best_model.save(save_path)
            if self.verbose > 0:
                print(f"Best model saved with mean reward at {save_path}")
