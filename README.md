# Shepherding Environment

## Description
This environment simulates a shepherding scenario where herders need to guide targets towards a goal region while avoiding obstacles and repelling the targets from getting too close. It is designed for use with reinforcement learning algorithms to train agents to effectively shepherd the targets.

## Installation
To install the Shepherding Environment, follow these steps:
1. Clone the repository: `git clone https://github.com/stefanocovone/shepherding.git`
2. Navigate to the repository directory: `cd shepherding`
3. Activate your Python/Conda environment, e.g.: `conda activate myenv`
5. Install the environment in editable mode: `pip install -e .`

## Usage
Once installed, you can use the Shepherding Environment in your Python code as follows:

```python
import gym
import shepherding

# Create the environment
env = gym.make('Shepherding-v0')

# Reset the environment
observation = env.reset()

# Perform actions and interact with the environment
done = False
while not done:
    action = ...  # Your action selection logic here
    observation, reward, done, info = env.step(action)
    # Your code to handle observations, rewards, and termination

# Close the environment
env.close()
