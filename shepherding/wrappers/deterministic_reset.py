from gymnasium import Wrapper
import numpy as np

from typing import Optional


class DeterministicReset(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        radius = 6*np.sqrt(2)
        angle = self.np_random.uniform(0, 2 * np.pi)
        # angle = np.pi/4

        target_pos = [radius * np.cos(angle), radius * np.sin(angle)]
        herder_pos = [(radius+1) * np.cos(angle*0.5), (radius+1) * np.sin(angle*0.5)]

        self.unwrapped.herder_pos[:] = np.array(herder_pos)
        self.unwrapped.target_pos[:] = np.array(target_pos)

        observation = self.unwrapped._get_obs()
        info = self.unwrapped._get_info()

        if self.unwrapped.render_mode == "human":
            self.unwrapped._render_frame()

        return observation, {}