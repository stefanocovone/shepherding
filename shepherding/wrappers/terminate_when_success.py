from gymnasium import Wrapper


class TerminateWhenSuccessful(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info['fraction_captured_targets'] == 1:
            terminated = True

        return obs, reward, terminated, truncated, info
