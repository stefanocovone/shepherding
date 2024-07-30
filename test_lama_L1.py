import os
import numpy as np
import gymnasium as gym
from shepherding.wrappers import TerminateWhenSuccessful
from shepherding.utils.control_rules import herder_actions

import torch


# Function to reshape tensor as in the validate function
def reshape_tensor(tensor, num_envs, max_episode_steps):
    num_steps, envs = tensor.shape[0], tensor.shape[1]
    reshaped_tensor = tensor.view(-1, max_episode_steps, envs, *tensor.shape[2:])
    return reshaped_tensor


# Parameters
parameters = {
    'num_herders': 1,
    'num_targets': 1,
    'noise_strength': 1,
    'rho_g': 5,
    'region_length': 50,
    'xi': 1000,
    'dt': 0.05,
    'k_T': 3,
    'k_rep': 100,
    'simulation_dt': 0.01,
    'solver': 'Euler',
}
env_id = 'Shepherding-v0'
num_episodes = 1000
max_episode_steps = 1200
num_envs = 1  # Number of parallel environments
device = 'cpu'

# Initialize environment
env = gym.make(env_id, render_mode='rgb_array', parameters=parameters)
env._max_episode_steps = max_episode_steps

# Initialize data storage
observations = np.zeros((num_episodes * max_episode_steps, num_envs,4))
cumulative_rewards = np.zeros((num_episodes * max_episode_steps, num_envs))
control_actions = np.zeros((num_episodes * max_episode_steps, num_envs) + env.action_space.shape)

step = 0
episode = 0

while episode < num_episodes:
    observation, info = env.reset(seed=episode)
    episode_step = 0
    done = False
    cum_reward = 0
    chi = 0

    while not done:
        # Choose an action (using herder_actions as a placeholder)
        action = herder_actions(env, observation)

        target_pos = env.unwrapped.target_pos
        herder_pos = env.unwrapped.herder_pos
        diff_pos = - herder_pos + target_pos

        obs = np.array([diff_pos, target_pos]) / env.unwrapped.region_length

        # Save observation and action
        observations[step] = obs.flatten()
        control_actions[step] = action

        # Step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Save reward
        cumulative_rewards[step] = reward

        step += 1
        episode_step += 1
        done = terminated or truncated
        cum_reward += reward
        chi = max(info["fraction_captured_targets"], chi)

    print(f"episode: {episode + 1}, chi: {chi}, success: {terminated}, reward: {cum_reward}")
    episode += 1

# Reshape and save the data
observations = reshape_tensor(torch.tensor(observations), num_envs, max_episode_steps).numpy()[:num_episodes]
control_actions = reshape_tensor(torch.tensor(control_actions), num_envs, max_episode_steps).numpy()[:num_episodes]
cumulative_rewards = reshape_tensor(torch.tensor(cumulative_rewards).unsqueeze(dim=-1), num_envs,
                                    max_episode_steps).numpy()[:num_episodes].squeeze().sum(axis=1)

save_path = f"runs/{env_id}__Lama_1/{env_id}__Lama_1_validation.npz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.savez(save_path,
         cumulative_rewards=cumulative_rewards,
         observations=observations,
         control_actions=control_actions)

# Close the environment
env.close()
