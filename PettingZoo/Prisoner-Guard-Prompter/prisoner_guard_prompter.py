import numpy as np
from copy import copy
import functools, pygame
import pygame.gfxdraw
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

GREY = (70, 70, 70)
DARKGREY = (50, 50, 50)
LIGHTGREY = (80, 80, 80)
RED = (173, 78, 78)
GREEN = (78, 173, 78)
ORANGE = (207, 135, 68)

class MovableAgent:
    def __init__(self) -> None:
        self.position = np.empty(2, dtype=np.float32)
        self.velocity = np.empty(2, dtype=np.float32)
        self.mass = 1.0
        self.max_velocity = 2.0

    def accelerate(self, force, dt) -> None:
        accel = force / self.mass
        self.velocity += accel * dt
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)

    def move(self, dt) -> None:
        self.position += self.velocity * dt


class Prisoner(MovableAgent):
    def __init__(self, length) -> None:
        super().__init__()
        self.message = np.zeros(length, dtype=np.float32)
        self.name = "prisoner"


class Guard(MovableAgent):
    def __init__(self) -> None:
        super().__init__()
        self.name = "guard"
        self.max_velocity = 1.6


class Prompter:
    def __init__(self, length) -> None:
        self.message = np.zeros(length, dtype=np.float32)
        self.position = np.empty(2, dtype=np.float32)
        self.name = "prompter"

class Escape:
    def __init__(self) -> None:
        self.position = np.empty(2, dtype=np.float32)
        self.name = "escape"


def env(*args, **kwargs):
    return raw_env(*args, **kwargs)


class raw_env(ParallelEnv):
    metadata = {
        "name": "prisoner-guard-prompter",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    def __init__(self, channel_length=8, max_cycles=150, 
                 render_mode=None, screen_size=640):
        self.channel_length = channel_length
        self.prisoner = Prisoner(channel_length)
        self.guard = Guard()
        self.prompter = Prompter(channel_length)
        self.escape = Escape()
        self.dt = 0.1
        self.possible_agents = [self.prisoner.name, 
                                self.prompter.name]
        self.dist_thres = 0.5
        self.max_cycles = max_cycles
        self.timestep = None
        self.observation_spaces = dict()
        self.action_spaces = dict()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # validation screen size
        s = screen_size
        assert isinstance(s, (int, tuple, list, np.ndarray)), "Incorrect screen size"
        if isinstance(s, int):
            assert s > 100 and s < 4000, "Incorrect screen size"
            self.screen_size = (s, s)
        elif isinstance(s, (tuple, list, np.ndarray)):
            assert len(s) == 2, "Incorrect screen size"
            assert s[0] > 100 and s[0] < 4000 and s[1] > 100 and s[1] < 4000, "Incorrect range for screen size"
            self.screen_size = int(s[0]), int(s[1])
        self.scale = None
        
    def reset(self, seed=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        if seed:
            np.random.seed(seed)
        self.prisoner.position = np.random.uniform(-15, 15, 2).astype(np.float32)
        self.guard.position =    np.random.uniform(-15, 15, 2).astype(np.float32)
        self.prompter.position = np.random.uniform(-15, 15, 2).astype(np.float32)
        self.escape.position =   np.random.uniform(-15, 15, 2).astype(np.float32)
        self.prisoner.velocity = np.random.uniform( -1,  1, 2).astype(np.float32)
        self.guard.velocity =    np.random.uniform( -1,  1, 2).astype(np.float32)
        self.scale = min(self.screen_size) / 50.0
        
        self.init_spaces()
        observations = self.get_observations()
        infos = {self.prisoner.name: {}, self.prompter.name: {}}
        return observations, infos
    
    def init_spaces(self):
        # observation space for prisoner:
        length = 4 + self.prompter.message.shape[0]
        prisoner_os = Box(low=-np.inf, high=np.inf, shape=(length,), dtype=np.float32)
        # observation space for prompter:
        prompter_os = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # action space for prompter:
        prompter_as = Box(low=0.0, high=1.0, shape=(length,), dtype=np.float32)
        # collect spaces:
        self.observation_spaces[self.prisoner.name] = prisoner_os
        self.observation_spaces[self.prompter.name] = prompter_os
        self.action_spaces[self.prisoner.name] = Discrete(5)
        self.action_spaces[self.prompter.name] = prompter_as

    def get_observations(self):
        observations = {
            self.prisoner.name: np.concatenate((
                                self.prisoner.velocity, 
                                self.prompter.position - self.prisoner.position,
                                self.prompter.message,
                                )),
            self.prompter.name: np.concatenate((
                                self.guard.position - self.prompter.position,
                                self.escape.position - self.prompter.position,
                                )) 
        }
        return observations
    
    def step(self, actions):
        prisoner_action = actions["prisoner"]
        prompter_action = actions["prompter"]

        if prisoner_action == 1:
            self.prisoner.accelerate(force=np.array([-1,  0]), dt=self.dt)
        elif prisoner_action == 2:
            self.prisoner.accelerate(force=np.array([ 1,  0]), dt=self.dt)
        elif prisoner_action == 3:
            self.prisoner.accelerate(force=np.array([ 0, -1]), dt=self.dt)
        elif prisoner_action == 4:
            self.prisoner.accelerate(force=np.array([ 0,  1]), dt=self.dt)
        self.prisoner.move(self.dt)
        
        self.prompter.message = prompter_action

        force = np.clip(self.prisoner.position - self.guard.position, -1, 1)
        self.guard.accelerate(force, dt=self.dt)
        self.guard.move(dt=self.dt)

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        if self.distance(self.prisoner, self.escape) < self.dist_thres:
            rewards[self.prisoner.name] = 1 
            rewards[self.prompter.name] = 1
            rewards[self.guard.name] = -1
            terminations = {a: True for a in self.agents}
            self.agents = []
        if self.distance(self.prisoner, self.guard) < self.dist_thres:
            rewards[self.prisoner.name] = -1 
            rewards[self.prompter.name] = -1
            rewards[self.guard.name] = 1
            terminations = {a: True for a in self.agents}
            self.agents = []
        if self.timestep > self.max_cycles:
            truncations = {a: True for a in self.agents}
            self.agents = []
        self.timestep += 1

        if self.render_mode == "human":
            self.render(static=False)
        
        if any(terminations.values()) or all(truncations.values()):
            print("Game Over!")
            self.agents = []
            self.render(static=True)

        observations = self.get_observations()
        infos = {self.prisoner.name: {}, self.prompter.name: {}}
        return observations, rewards, terminations, truncations, infos

    @staticmethod
    def distance(entity_1, entity_2):
        return np.sqrt(np.sum((entity_1.position - entity_2.position)**2))

    def render(self, static=True):
        if self.render_mode is None:
            from gymnasium.logger import warn
            warn("You are calling render method without specifying any render mode. "
                 "You can specify the render_mode at initialization, "
                 'e.g. render_mode="rgb_array" or render_mode="rgb_array"')
            return
        
        # self.screen_size = (self.screen_size // self.grid_size) * self.grid_size
        if self.render_mode == "human":
            self.human_render(static, "Prisoner Guard Prompter")
        elif self.render_mode == "rgb_array":
            return self.array_render()
    
    def human_render(self, static=False, caption=""):
        if self.screen is None:
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption(caption)

        if static:
            running = True
            while running:
                self.draw()
                self.clock.tick(self.metadata["render_fps"])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        self.close()
                    self.event_handler(event)  
        else:
            self.draw()
            self.clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close(truncate=True)
                self.event_handler(event)    

    def event_handler(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                self.scale *= 1.5
            elif event.button == 5:
                self.scale *= 0.7

    
    def array_render(self):
        self.screen = pygame.Surface((self.screen_size, self.screen_size))
        self.draw()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), 
            axes=(1, 0, 2))
    
    def get_coord_for_display(self, agent):
        shift_x = self.screen_size[0] // 2
        shift_y = self.screen_size[1] // 2
        shift = np.array([shift_x, shift_y])
        return (agent.position * self.scale).astype(int) + shift 

    def draw(self):
        self.screen.fill(GREY)
        radius = int(self.scale) // 2
        pygame.draw.circle(self.screen, GREEN, 
                           self.get_coord_for_display(self.prisoner), radius)
        pygame.draw.circle(self.screen, RED, 
                           self.get_coord_for_display(self.guard), radius)
        pygame.draw.circle(self.screen, ORANGE, 
                           self.get_coord_for_display(self.prompter), radius)
        width = 2 * radius
        position = self.get_coord_for_display(self.escape) - radius
        box = (position[0], position[1], width, width)
        pygame.draw.rect(self.screen, DARKGREY, box)
        pygame.display.flip()

    def close(self, truncate=False):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            if truncate:
                self.agents = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]


if __name__ == "__main__":
    parallel_env = raw_env(render_mode="human", max_cycles=1000)
    # parallel_api_test(parallel_env, num_cycles=1_000_000)
    observations, infos = parallel_env.reset()
    while parallel_env.agents:
        actions = {agent: parallel_env.action_space(agent).sample() 
                   for agent in parallel_env.agents}
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    print(observations)
    print(rewards)
    print(terminations)
    print(truncations)