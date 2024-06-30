import gymnasium as gym
from shepherding.wrappers import TerminateWhenSuccessful
from shepherding.utils.control_rules import herder_actions


parameters = {
    'num_herders': 20,
    'num_targets': 20*20,
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
env = gym.make('Shepherding-v0', render_mode='rgb_array', parameters=parameters)
env._max_episode_steps = 100000
env = TerminateWhenSuccessful(env)

# Run the simulation for a certain number of steps
truncated = False
terminated = False

for episode in range(1, 50 + 1):
    # Reset the environment to get the initial observation
    observation, info = env.reset(seed=episode)
    step = 0
    cum_reward = 0
    chi = 0
    truncated = False
    terminated = False
    while not (terminated or truncated):
        step += 1
        # Choose a random action (here, randomly setting velocities for herders)
        action = herder_actions(env, observation)
        # Take a episode_step in the environment by applying the chosen action
        observation, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        chi = max(info["fraction_captured_targets"], chi)

    print("episode: ", episode, "chi: ", chi, "success: ", terminated, "reward: ", cum_reward)

# Close the environment (optional)
env.close()
