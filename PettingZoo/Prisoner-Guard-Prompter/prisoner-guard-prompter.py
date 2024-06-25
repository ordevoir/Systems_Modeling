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
        self.max_velocity = 1.0


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
    def __init__(self, channel_length=8, max_cycles=150, render_mode=None):
        self.channel_length = channel_length
        self.prisoner = Prisoner(channel_length)
        self.guard = Guard()
        self.prompter = Prompter(channel_length)
        self.escape = Escape()
        self.dt = 0.1
        self.possible_agents = [self.prisoner.name, 
                                self.guard.name,
                                self.prompter.name]
        self.dist_thres = 0.1
        self.max_cycles = max_cycles
        self.timestep = None

    def reset(self, seed=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        if seed:
            np.random.seed(seed)
        self.prisoner.position = np.random.uniform(-10, 10, 2).astype(np.float32)
        self.guard.position =    np.random.uniform(-10, 10, 2).astype(np.float32)
        self.prompter.position = np.random.uniform(-10, 10, 2).astype(np.float32)
        self.escape.position =   np.random.uniform(-10, 10, 2).astype(np.float32)
        self.prisoner.velocity = np.random.uniform( -2,  2, 2).astype(np.float32)
        self.guard.velocity =    np.random.uniform( -2,  2, 2).astype(np.float32)

        observations = self.get_observations()
        infos = {self.prisoner.name: {}, self.prompter.name: {}}
        return observations, infos
    
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
            self.prisoner.accelerate(force=(-1,  0), dt=self.dt)
        elif prisoner_action == 2:
            self.prisoner.accelerate(force=( 1,  0), dt=self.dt)
        elif prisoner_action == 3:
            self.prisoner.accelerate(force=( 0, -1), dt=self.dt)
        elif prisoner_action == 4:
            self.prisoner.accelerate(force=( 0,  1), dt=self.dt)
        self.prisoner.move(self.dt)
        
        self.prompter.message = prompter_action

        force = np.clip(self.prisoner.position - self.guard.position, -1, 1)
        self.guard.accelerate(force, dt=self.dt)
        self.guard.move(dt=self.dt)

        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        if self.distance(self.prisoner, self.escape) < self.dist_thres:
            rewards[self.prisoner.name] = 1 
            rewards[self.prompter.name] = 1
            rewards[self.guard.name] = -1
            terminations = {a: True for a in self.agents}

        if self.distance(self.prisoner, self.guard) < self.dist_thres:
            rewards[self.prisoner.name] = -1 
            rewards[self.prompter.name] = -1
            rewards[self.guard.name] = 1
            terminations = {a: True for a in self.agents}

        truncations = {a: False for a in self.agents}
        if self.timestep > self.max_cycles:
            truncations = {a: True for a in self.agents}
            self.agents = []
        self.timestep += 1

        observations = self.get_observations()
        infos = {self.prisoner.name: {}, self.prompter.name: {}}
        return observations, rewards, terminations, truncations, infos

    @staticmethod
    def distance(entity_1, entity_2):
        return np.sqrt(np.sum((entity_1.position - entity_2.position)**2))

    # def render(self)



if __name__ == "__main__":
    parallel_env = raw_env()
    # parallel_api_test(parallel_env, num_cycles=1_000_000)
    observations, infos = parallel_env.reset(seed=42)
    while parallel_env.agents:
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    parallel_env.close()