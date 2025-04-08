import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class CartPoleILQREnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.max_force = 20.0
        self.tau = 0.02

        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.steps_beyond_done = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {}

    def _state_eq(self, st, u):
        x, x_dot, theta, theta_dot = st
        force = u[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.5 * np.pi, 0.0], dtype=np.float32)
        self.steps_beyond_done = None
        return self._get_obs(), self._get_info()

    def step(self, action):
        # assert self.action_space.contains(action)
        self.state = self._state_eq(self.state, action)
        x, x_dot, theta, theta_dot = self.state

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not terminated else 0.0

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode is None:
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0

        # Draw track
        pygame.draw.line(self.surf, (0, 0, 0), (0, carty), (screen_width, carty), 2)

        # Draw cart
        cart_rect = pygame.Rect(0, 0, cartwidth, cartheight)
        cart_rect.center = (cartx, carty)
        pygame.draw.rect(self.surf, (0, 0, 255), cart_rect)

        # Draw pole
        pole_x = cartx
        pole_y = carty
        l = polelen
        angle = -x[2]
        x_tip = pole_x + l * np.sin(angle)
        y_tip = pole_y - l * np.cos(angle)
        pygame.draw.line(self.surf, (255, 0, 0), (pole_x, pole_y), (x_tip, y_tip), int(polewidth))

        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

if __name__ == '__main__':
    env = CartPoleILQREnv(render_mode="human")
    obs, _ = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()

    env.close()
