import argparse
import gymnasium as gym
import numpy as np
from shepherding.wrappers import LowLevelPPOPolicy

import cProfile
import pstats
import os

def herder_actions(env, observation, random=False):
    if random:
        actions = np.random.uniform(-1, 1, size=env.action_space.shape)
    else:

        # Assuming observation is a numpy array and env.num_herders is the number of herders
        # xi is the distance threshold, v_h is the herder velocity scalar, alpha and delta are given parameters
        xi = env.xi
        v_h = env.herder_max_vel
        alpha = env.alpha
        delta = env.lmbda / 2
        rho_g = env.rho_g

        # Extract herder and target positions from the observation
        herder_pos = observation[0:env.num_herders, :]  # Shape (N, 2)
        target_pos = observation[env.num_herders:, :]  # Shape (M, 2)

        # Expand dimensions of herder_pos and target_pos to enable broadcasting
        herder_pos_exp = herder_pos[:, np.newaxis, :]  # Shape (N, 1, 2)
        target_pos_exp = target_pos[np.newaxis, :, :]  # Shape (1, M, 2)

        # Calculate the relative positions
        relative_positions = target_pos_exp - herder_pos_exp  # Shape (N, M, 2)

        # Compute the Euclidean distances between herders and targets
        distances = np.linalg.norm(relative_positions, axis=2)  # Shape (N, M)

        # Find the index of the closest herder for each target
        closest_herders = np.argmin(distances, axis=0)  # Shape (M,)

        # Create a boolean mask where each target is only considered if it's closer to the current herder
        closest_mask = np.zeros_like(distances, dtype=bool)
        np.put_along_axis(closest_mask, closest_herders[np.newaxis, :], True, axis=0)

        # Create a boolean mask where distances are less than xi and the herder is the closest one
        mask = (distances < xi) & closest_mask  # Shape (N, M)

        # Calculate the absolute distances from the origin for the targets
        absolute_distances = np.linalg.norm(target_pos, axis=1)  # Shape (M,)

        # Use broadcasting to expand the absolute distances to match the shape of the mask
        expanded_absolute_distances = np.tile(absolute_distances, (env.num_herders, 1))  # Shape (N, M)

        # Apply the mask to get valid distances only
        valid_absolute_distances = np.where(mask, expanded_absolute_distances, -np.inf)  # Shape (N, M)

        # Find the index of the target with the maximum absolute distance from the origin for each herder
        selected_target_indices = np.argmax(valid_absolute_distances, axis=1)  # Shape (N,)

        # Create a mask to identify herders that have no valid targets
        no_valid_target_mask = np.all(~mask, axis=1)

        # Replace invalid indices with -1 (indicating no target)
        selected_target_indices = np.where(no_valid_target_mask, -1, selected_target_indices)

        # Create a vector (N, 2) to store the absolute position of the selected target for each herder
        selected_target_positions = np.zeros((env.num_herders, 2))
        selected_target_positions[~no_valid_target_mask] = target_pos[selected_target_indices[~no_valid_target_mask]]

        # Calculate unit vectors for herders and selected targets
        herder_unit_vectors = herder_pos / np.linalg.norm(herder_pos, axis=1, keepdims=True)  # Shape (N, 2)
        selected_target_unit_vectors = np.zeros((env.num_herders, 2))
        selected_target_unit_vectors[~no_valid_target_mask] = (
                target_pos[selected_target_indices[~no_valid_target_mask]] / np.linalg.norm(
            target_pos[selected_target_indices[~no_valid_target_mask]], axis=1, keepdims=True
        )
        )

        # Calculate actions for each herder
        actions = np.zeros((env.num_herders, 2))
        herder_abs_distances = np.linalg.norm(herder_pos, axis=1)  # Absolute distances of herders from the origin

        # If no target is selected and the herder's distance is less than rho_g, action is zero
        # Otherwise, action is v_h * herder_unit_vector
        no_target_selected = no_valid_target_mask & (herder_abs_distances < rho_g)
        actions[no_valid_target_mask & ~no_target_selected] = - v_h * herder_unit_vectors[
            no_valid_target_mask & ~no_target_selected]

        # If a target is selected, action is
        # alpha * (herder_pos - (selected_target_pos + delta * selected_target_unit_vector))
        actions[~no_valid_target_mask] = - alpha * (
                herder_pos[~no_valid_target_mask] - (
                selected_target_positions[~no_valid_target_mask] +
                delta * selected_target_unit_vectors[~no_valid_target_mask]
        )
        )

    return selected_target_indices


def run_simulation(env_name, params, solver, simulation_dt):
    profiler = cProfile.Profile()
    profiler.enable()

    env = gym.make(env_name, render_mode='rgb_array', parameters=params)
    env._max_episode_steps = 3000
    env = LowLevelPPOPolicy(env, 1)

    observations = []
    rewards = []

    for episode in range(1, 10 + 1):
        observation, info = env.reset(seed=episode)
        action = env.action_space.sample()
        step = 0
        cum_reward = 0
        truncated = False
        terminated = False

        episode_observations = []

        while not (terminated or truncated):
            step += 1
            action = herder_actions(env, observation, random=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_observations.append(observation)
            cum_reward += reward

        observations.append(episode_observations)
        rewards.append(cum_reward)
        print(f"episode: {episode}, reward: {cum_reward}")

    env.close()

    profiler.disable()
    stats = pstats.Stats(profiler)
    total_time = stats.total_tt
    stats.sort_stats('tottime').print_stats(10)

    os.makedirs("simulations", exist_ok=True)
    filename = f"simulations/{env_name}_{solver}_{simulation_dt}_{params['num_herders']}_{params['num_targets']}.npz"
    np.savez(filename, observations=observations, rewards=rewards, total_time=total_time)

if __name__ == "__main__":
    configurations = [
        {'num_herders': 1, 'num_targets': 1},
        {'num_herders': 5, 'num_targets': 20},
        {'num_herders': 10, 'num_targets': 50},
        {'num_herders': 20, 'num_targets': 100},
        {'num_herders': 40, 'num_targets': 200},
    ]

    solvers = ['Euler', 'SRI2']
    simulation_dts = [0.05, 0.01, 0.005, 0.001]

    parser = argparse.ArgumentParser(description='Shepherding Simulation Runner')
    parser.add_argument('--noise_strength', type=float, default=1, help='Noise strength')
    parser.add_argument('--rho_g', type=float, default=5, help='Radius of the goal region')
    parser.add_argument('--region_length', type=float, default=60, help='Length of the region')
    parser.add_argument('--xi', type=float, default=1000, help='Distance threshold')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step')
    parser.add_argument('--k_T', type=float, default=3, help='Parameter k_T')
    parser.add_argument('--k_rep', type=float, default=100, help='Parameter k_rep')

    args = parser.parse_args()
    base_params = vars(args)

    # Run Shepherding-v0 with Euler solver and 0.05 simulation_dt for all configurations
    env_name = 'Shepherding-v0'
    for config in configurations:
        params = {**base_params, **config, 'solver': 'Euler', 'simulation_dt': 0.05}
        print(f"Running simulation with {params}")
        run_simulation(env_name, params, 'Euler', 0.05)

    # Run Shepherding-v1 with the specified configurations, solvers, and simulation_dts
    env_name = 'Shepherding-v1'
    for config in configurations:
        for solver in solvers:
            for simulation_dt in simulation_dts:
                params = {**base_params, **config, 'solver': solver, 'simulation_dt': simulation_dt}
                print(f"Running simulation with {params}")
                run_simulation(env_name, params, solver, simulation_dt)