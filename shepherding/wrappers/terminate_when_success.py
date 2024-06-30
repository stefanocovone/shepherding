from gymnasium import Wrapper


class TerminateWhenSuccessful(Wrapper):
    def __init__(self, env, num_steps = 1):
        super().__init__(env)
        self.env = env
        self.success_buffer = 0
        self.num_steps = num_steps

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info['fraction_captured_targets'] == 1:
            self.success_buffer += 1
        else:
            self.success_buffer = 0

        if self.success_buffer >= self.num_steps:
            terminated = True

        return obs, reward, terminated, truncated, info
