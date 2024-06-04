import numpy as np
from gymnasium import ActionWrapper, spaces


class FlattenAction(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_action_shape = self.env.action_space.shape
        assert len(original_action_shape) == 2 and original_action_shape[1] == 2, \
            "Original action space shape must be (N, 2)"
        N = original_action_shape[0]

        # Define the new action space with shape (2N, 1)
        low = self.env.action_space.low.flatten().reshape(2 * N)
        high = self.env.action_space.high.flatten().reshape(2 * N)
        self.action_space = spaces.Box(low=low, high=high, dtype=self.env.action_space.dtype)

    def action(self, action):
        N = int(self.env.action_space.shape[0] / 2)
        return np.reshape(action, (self.env.num_herders, 2))  # Reshape to (N, 2)