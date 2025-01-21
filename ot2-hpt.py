from stable_baselines3 import PPO
import os
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from ot2_wrapper_final import OT2Env
from clearml import Task
import typing_extensions
import tensorboard
import sys
from save_best_callback import SaveBestRewardAtEndCallback


os.environ['WANDB_API_KEY'] = 'f0c26550ec9902de91ecb9f54fbbdfc3c6bbb24e'

task = Task.init(project_name="Mentor Group E/Group DMRM", task_name="ppo-hpt_michal")

# Define sweep config
sweep_config = {
    "method": "bayes",
    "name": "sweep_michal",
    "metric": {"goal": "minimize", "name": "rollout/ep_len_mean"},
    "parameters": {
        #"learning_rate": {"values": [0.0003, 0.0001, 0.00005, 0.0005, 0.00008]},
        "n_steps": {"distribution": "int_uniform", "min": 2048, "max": 4096},
        # "batch_size": {"distribution": "int_uniform", "min": 32, "max": 256},
        # "gamma": {"distribution": "uniform", "min": 0.9, "max": 0.999},
    },
}

# Connect the dictionary to your CLEARML Task
parameters_dict = Task.current_task().connect(sweep_config)

#setting the base docker image
#task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
#task.execute_remotely(queue_name="default")

sweep_id = wandb.sweep(sweep_config, project="sweep_for_weights")

def main(config=None):
    run = wandb.init(config, sync_tensorboard=True)

    config = run.config

    n_steps = config.n_steps 

    print(f"Starting training with n_steps={n_steps}")

    env = OT2Env()
    env.reset(seed=42)

    model = PPO("MlpPolicy", env, n_steps=n_steps, verbose=1, tensorboard_log="./logs_final_hpt")

    # Use the callback to save the best model only at the end
    model.learn(total_timesteps=2_000_000, reset_num_timesteps=False, callback=SaveBestRewardAtEndCallback(log_dir="./logs_final_hpt"))

    # Save the trained model locally
    model.save("michal_model.zip")
    # Save the model to W&B
    run.save("michal_model.zip")

    run.finish()


wandb.agent(sweep_id, main)