import numpy as np
from copy import copy

class MovableAgent:
    def __init__(self) -> None:
        self.position = np.empty(2, dtype=np.float32)
        self.velocity = np.empty(2, dtype=np.float32)
        self.mass = 1.0
        self.max_velocity = 2.0

    def accelerate(self, force, dt) -> None:
        accel = force / self.mass
        self.velosity += accel * dt
        self.velosity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)

    def move(self, dt) -> None:
        self.position += self.velosity * dt


class Prisoner(MovableAgent):
    def __init__(self, length) -> None:
        super().__init__()
        self.message = np.zeros(length, dtype=np.float32)
        self.name = "prisoner"


class Guard(MovableAgent):
    def __init__(self) -> None:
        super().__init__()
        self.name = "guard"


class Prompter:
    def __init__(self, length) -> None:
        self.message = np.zeros(length, dtype=np.float32)
        self.position = np.empty(2, dtype=np.float32)
        self.name = "prompter"

class Escape:
    def __init__(self) -> None:
        self.position = np.empty(2, dtype=np.float32)
        self.name = "escape"


class raw_env():
    def __init__(self, channel_length=8):
        self.channel_length = channel_length
        self.prisoner = Prisoner(channel_length)
        self.guard = Guard()
        self.prompter = Prompter(channel_length)
        self.escape = Escape()
        self.possible_agents = [self.prisoner.name, 
                                self.guard.name,
                                self.prompter.name]
        
    def reset(self, seed=None):
        self.agens = copy(self.possible_agents)
        if seed:
            np.random.seed(seed)
        self.prisoner.position = np.random.uniform(-10, 10, 2).astype(np.float32)
        self.guard.position =    np.random.uniform(-10, 10, 2).astype(np.float32)
        self.prompter.position = np.random.uniform(-10, 10, 2).astype(np.float32)
        self.escape.position =   np.random.uniform(-10, 10, 2).astype(np.float32)
        self.prisoner.velocity = np.random.uniform( -2,  2, 2).astype(np.float32)
        self.guard.velocity =    np.random.uniform( -2,  2, 2).astype(np.float32)

        observations = {
            self.prisoner.name: np.concatenate(
                                self.prisoner.velocity, 
                                self.prompter.position - self.prisoner.position,
                                self.prompter.message,
                                ),
            self.prompter.name: np.concatenate(
                                self.guard.position - self.prompter.position,
                                self.escape.position - self.prompter.position,
                                ) 
        }
        infos = {self.prisoner.name: {}, self.prompter.name: {}}
        return observations, infos
    
    def step(self, actions):
        prisoner_action = actions["prisoner"]
        prompter_action = actions["prompter"]
