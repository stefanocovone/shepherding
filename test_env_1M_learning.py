import gymnasium as gym
from shepherding.wrappers import MultiAgentPPO
from shepherding.utils.control_rules import select_targets
from shepherding.utils import ActorCriticDiscrete

parameters = {
    'num_herders': 2,
    'num_targets': 5,
    'num_targets_min': 2,
    'num_targets_max': 5,
    'noise_strength': .1,
    'rho_g': 5,
    'region_length': 50,
    'xi': 1000,
    'dt': 0.05,
    'k_T': 3,
    'k_rep': 100,
    'simulation_dt': 0.001,
    'solver': 'Euler',
}
env = gym.make('Shepherding-v0', render_mode='human', parameters=parameters, rand_target=False)
env._max_episode_steps = 5000
env = MultiAgentPPO(env, 20)

select_action = ActorCriticDiscrete.ActorCritic()

# Run the simulation for a certain number of steps
truncated = False
terminated = False

for episode in range(1, 100 + 1):
    # Reset the environment to get the initial observation
    observation, info = env.reset()
    action = env.action_space.sample()  # Example action
    step = 0
    cum_reward = 0
    truncated = False
    terminated = False

    while not (terminated or truncated):

        step += 1
        # action = select_targets(env, observation)
        action = env.action_space.sample()  # Example action
        # Take a episode_step in the environment by applying the chosen action
        observation, reward, terminated, truncated, _ = env.step(action)
        cum_reward += reward

    print("episode: ", episode, "reward: ", cum_reward)

# Close the environment (optional)
env.close()
