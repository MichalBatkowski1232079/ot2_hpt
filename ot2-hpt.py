from stable_baselines3 import PPO
import wandb
import os
from ot2_wrapper_final import OT2Env
from save_best_callback import SaveBestRewardAtEndCallback

os.environ['WANDB_API_KEY'] = 'f0c26550ec9902de91ecb9f54fbbdfc3c6bbb24e'

# Define sweep config
sweep_config = {
    "method": "bayes",
    "name": "sweep_michal",
    "metric": {"goal": "minimize", "name": "rollout/ep_len_mean"},
    "parameters": {
        "n_steps": {"distribution": "int_uniform", "min": 2048, "max": 4096},
    },
}

sweep_id = wandb.sweep(sweep_config, project="sweep_for_weights")

def main(config=None):
    run = wandb.init(config, sync_tensorboard=True)

    config = run.config
    n_steps = config.n_steps  # Get current n_steps from sweep

    print(f"Starting training with n_steps={n_steps}")

    env = OT2Env()
    env.reset(seed=42)

    log_dir = "./logs_final_hpt"
    callback = SaveBestRewardAtEndCallback(log_dir=log_dir, n_steps=n_steps)  # Only include n_steps in filename

    model = PPO("MlpPolicy", env, n_steps=n_steps, verbose=1, tensorboard_log=log_dir)

    # Train the model
    model.learn(total_timesteps=2_000_000, reset_num_timesteps=False, callback=callback)

    # Save the final model locally with n_steps in the filename
    final_model_filename = f"final_model_nsteps_{n_steps}.zip"
    model.save(final_model_filename)
    print(f"Final model saved as {final_model_filename}")

    # Save the final model to W&B
    run.save(final_model_filename)
    print(f"Final model uploaded to W&B as {final_model_filename}")

    run.finish()

wandb.agent(sweep_id, main)
