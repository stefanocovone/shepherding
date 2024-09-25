import gymnasium as gym
from shepherding.wrappers import LowLevelPPOPolicy
from shepherding.utils.control_rules import select_targets, select_targets_learning
from shepherding.utils import ActorCriticDiscrete
from gymnasium.wrappers import RecordVideo

import os
import numpy as np
import torch

# Function to reshape tensor as in the validate function
def reshape_tensor(tensor, num_envs, max_episode_steps):
    num_steps, envs = tensor.shape[0], tensor.shape[1]
    reshaped_tensor = tensor.view(-1, max_episode_steps, envs, *tensor.shape[2:])
    return reshaped_tensor

use_learning = True

parameters = {
    'num_herders': 2,
    'num_targets': 7,
    'num_targets_min': 2,
    'num_targets_max': 7,
    'noise_strength': .1,
    'rho_g': 5,
    'region_length': 50,
    'xi': 2000,
    'dt': 0.05,
    'k_T': 3,
    'k_rep': 100,
    'simulation_dt': 0.001,
    'solver': 'Euler',
}

env_id = 'Shepherding-v0'
num_episodes = 1000
max_episode_steps = 2000
num_envs = 1  # Number of parallel environments
device = 'cpu'

env = gym.make('Shepherding-v0', render_mode='human', parameters=parameters, rand_target=False)
env._max_episode_steps = max_episode_steps
# env = RecordVideo(env, video_folder="videos")
env = LowLevelPPOPolicy(env, 20)

model = ActorCriticDiscrete.ActorCritic()


# Initialize data storage
observations = np.zeros((num_episodes * max_episode_steps, num_envs, 18))
cumulative_rewards = np.zeros((int(num_episodes * max_episode_steps / 20), num_envs))
control_actions = np.zeros((int(num_episodes * max_episode_steps / 20), num_envs) + env.action_space.shape)


step = 0
episode = 0


while episode < num_episodes:
    observation, info = env.reset(seed=episode)
    episode_step = 0
    done = False
    cum_reward = 0
    chi = 0

    while not done:
        # Choose an action
        action = select_targets(env, observation)

        # Step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Save observation and action
        observations[step:step+20] = info['obs_batch'].reshape(20,1,18)
        control_actions[int(step/20)] = action

        # Save reward
        cumulative_rewards[int(step/20)] = reward

        step += 20
        episode_step += 20
        done = terminated or truncated
        cum_reward += reward
        chi = max(info["fraction_captured_targets"], chi)

    print(f"episode: {episode + 1}, chi: {chi}, success: {terminated}, reward: {cum_reward}")
    episode += 1

# Reshape and save the data
observations = reshape_tensor(torch.tensor(observations), num_envs, max_episode_steps).numpy()[:num_episodes]
control_actions = reshape_tensor(torch.tensor(control_actions), num_envs, int(max_episode_steps/20)).numpy()[:num_episodes]
cumulative_rewards = reshape_tensor(torch.tensor(cumulative_rewards).unsqueeze(dim=-1), num_envs,
                                    int(max_episode_steps/20)).numpy()[:num_episodes].squeeze().sum(axis=1)

save_path = f"runs/{env_id}__Lama_2M_1/{env_id}__Lama_27_1_validation.npz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.savez(save_path,
         cumulative_rewards=cumulative_rewards,
         observations=observations,
         control_actions=control_actions)

# Close the environment
env.close()



