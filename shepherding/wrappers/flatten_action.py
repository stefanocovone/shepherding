import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, spaces

class FlattenAction(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return np.reshape(action, self.unwrapped.action_space.shape)