import random, functools
import numpy as np
from copy import copy
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete
import pygame

GREY = (70, 70, 70)
DARKGREY = (50, 50, 50)
LIGHTGREY = (80, 80, 80)
RED = (173, 78, 78)
GREEN = (78, 173, 78)

class raw_env(ParallelEnv):
    metadata = {
        "name": "simplest_env",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 3,
    }

    def __init__(self, 
                 grid_size=8, 
                 max_cycles=100, 
                 render_mode=None, 
                 screen_size=320):
        # entity positions:
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None

        self.grid_size = grid_size
        self.timestep = None
        self.max_cycles = max_cycles
        self.possible_agents = ["prisoner", "guard"]

        self.render_mode = render_mode
        self.screen_size = screen_size
        self.screen = None
        self.clock = None
    
    def reset(self, seed=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner_x = 0
        self.prisoner_y = 0

        self.guard_x = self.grid_size - 1
        self.guard_y = self.grid_size - 1

        self.escape_x = random.randint(2, self.grid_size-2)
        self.escape_y = random.randint(2, self.grid_size-2)

        observation = (
                self.prisoner_x + self.grid_size * self.prisoner_y,
                self.guard_x + self.grid_size * self.guard_y,
                self.escape_x + self.grid_size * self.escape_y,
            )
        observations = {
            "prisoner": {"observation": observation, "action_mask": [0, 1, 1, 0]},
            "guard": {"observation": observation, "action_mask": [1, 0, 0, 1]},
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        return observations, infos
    
    def step(self, actions):
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < self.grid_size-1:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < self.grid_size-1:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < self.grid_size-1:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < self.grid_size-1:
            self.guard_y += 1

        # Generate action masks
        prisoner_action_mask = np.ones(4, dtype=np.int8)
        if self.prisoner_x == 0:
            prisoner_action_mask[0] = 0  # Block left movement
        elif self.prisoner_x == self.grid_size-1:
            prisoner_action_mask[1] = 0  # Block right movement
        if self.prisoner_y == 0:
            prisoner_action_mask[2] = 0  # Block down movement
        elif self.prisoner_y == self.grid_size-1:
            prisoner_action_mask[3] = 0  # Block up movement

        guard_action_mask = np.ones(4, dtype=np.int8)
        if self.guard_x == 0:
            guard_action_mask[0] = 0
        elif self.guard_x == self.grid_size-1:
            guard_action_mask[1] = 0
        if self.guard_y == 0:
            guard_action_mask[2] = 0
        elif self.guard_y == self.grid_size-1:
            guard_action_mask[3] = 0

        # Action mask to prevent guard from going over escape cell
        if self.guard_x - 1 == self.escape_x:
            guard_action_mask[0] = 0
        elif self.guard_x + 1 == self.escape_x:
            guard_action_mask[1] = 0
        if self.guard_y - 1 == self.escape_y:
            guard_action_mask[2] = 0
        elif self.guard_y + 1 == self.escape_y:
            guard_action_mask[3] = 0

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > self.max_cycles:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observation = (
                self.prisoner_x + self.grid_size * self.prisoner_y,
                self.guard_x + self.grid_size * self.guard_y,
                self.escape_x + self.grid_size * self.escape_y,
            )
        observations = {
            "prisoner": {
                "observation": observation,
                "action_mask": prisoner_action_mask,
            },
            "guard": {
                "observation": observation, 
                "action_mask": guard_action_mask,
            },
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}
        
        if self.render_mode == "human":
            self.render(static=False)

        if any(terminations.values()) or all(truncations.values()):
            print("Game Over!")
            self.agents = []
            self.render(static=True)

        return observations, rewards, terminations, truncations, infos

    def render(self, static=True):
        if self.render_mode is None:
            from gymnasium.logger import warn
            warn("You are calling render method without specifying any render mode. "
                 "You can specify the render_mode at initialization, "
                 'e.g. render_mode="rgb_array" or render_mode="rgb_array"')
            return
        
        self.screen_size = (self.screen_size // self.grid_size) * self.grid_size
        if self.render_mode == "human":
            self.human_render(static)
        elif self.render_mode == "rgb_array":
            return self.array_render()


    def human_render(self, static=False):
        if self.screen is None:
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Simplest Environment")

        self.draw()
        pygame.display.flip()
        if static:
            running = True
            while running:
                self.clock.tick(self.metadata["render_fps"])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        self.close()
        else:
            self.clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close(truncate=True)

    def array_render(self):
        self.screen = pygame.Surface((self.screen_size, self.screen_size))
        self.draw()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), 
            axes=(1, 0, 2))

    def draw(self):
        self.screen.fill(GREY)
        length = self.screen_size
        step = length // (self.grid_size)
        radius = 0.4 * length // (self.grid_size)
        # draw guard and prisoner:
        red_circle_x = (self.guard_x) * step + step // 2
        red_circle_y = (self.guard_y) * step + step // 2
        pygame.draw.circle(self.screen, RED, (red_circle_x, red_circle_y), radius)
        green_circle_x = (self.prisoner_x) * step + step // 2
        green_circle_y = (self.prisoner_y) * step + step // 2
        pygame.draw.circle(self.screen, GREEN, (green_circle_x, green_circle_y), radius)
        # draw escape:
        blue_rect_x = self.escape_x * step
        blue_rect_y = self.escape_y * step
        pygame.draw.rect(self.screen, DARKGREY, (blue_rect_x, blue_rect_y, step, step))
        # draw grid:
        for i in range(1, self.grid_size):
            pygame.draw.line(self.screen, LIGHTGREY, (0, i*step), (length, i*step))
        for i in range(1, self.grid_size):
            pygame.draw.line(self.screen, LIGHTGREY, (i*step, 0), (i*step, length))

    def close(self, truncate=False):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            if truncate:
                self.agents = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([self.grid_size * self.grid_size] * 3)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)
    
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    parallel_env = raw_env(render_mode="human", grid_size=6)
    # parallel_api_test(parallel_env, num_cycles=1_000_000)
    observations, infos = parallel_env.reset(seed=42)
    while parallel_env.agents:
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    parallel_env.close()