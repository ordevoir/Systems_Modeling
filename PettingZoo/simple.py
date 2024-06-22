
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
from gymnasium.utils import seeding
import gymnasium
import pygame
import numpy as np

class Entity:
    def __init__(self):
        self.name = ""
        self.mass = 1.0
        self.pos = None
        self.vel = None
        self.com = None
        self.movable = False
        self.color = None

class Agent(Entity):
    def __init__(self):
        super().__init__()
        self.movable = True

# class AgentState

class Landmark(Entity):
    def __init__(self):
        super().__init__()

class World:
    def __init__(self):
        self.agents = []
        self.landmarks = []

        self.dim_c = 0
        self.dim_p = 2
        self.demping = 0.2

    @property
    def entities(self):
        return self.agents + self.landmarks

class BaseScenario:  # defines scenario upon which the world is built
    def make_world(self):  # create elements of the world
        raise NotImplementedError()

    def reset_world(self, world, np_random):  # create initial conditions of the world
        raise NotImplementedError()
    
class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        world.dim_c = 10
        num_agents = None
        num_landmarks = N
        
        world.agents = [Agent() for i in range(N)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"

        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.movable = False
        return world
    
    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.pos = np_random.uniform(-1, +1, world.dim_p)
            agent.vel = np.zeros(world.dim_p)
            agent.com = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        return 1
    
    def global_reward(self, world):
        return 1
    
    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.pos - agent.pos)
        com = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            com.append(other.com)
            other_pos.append(other.pos - agent.pos)
        return np.concatenate(
            [agent.vel, agent.pos], entity_pos, other_pos, com
        )


class raw_env(AECEnv):
    metadata = {
        "name": "hamilton_environment_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 20,
    }
    def __init__(self,
        render_mode=None,       
        ):
        self.render_mode = render_mode
        pygame.init()
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])

        self._seed()
        self.scenario = Scenario()      
        self.world = World()  
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }
        self._agent_selector = agent_selector(self.agents)
        self.action_spaces = dict()
        self.observation_spaces = dict()
        
        self.steps = 0
        self.current_actions = [None] * self.num_agents

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self):
        pass

    def step(self):
        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        
        self.enable_render()
        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def draw(self):
        
        self.screen.fill((255, 255, 255))       # clear screen

        # update bounds to center around agent
        all_poses = [entity.pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

from pettingzoo.utils import wrappers
def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        # if env.continuous_actions:
        #     env = wrappers.ClipOutOfBoundsWrapper(env)
        # else:
        # env = wrappers.AssertOutOfBoundsWrapper(env)
        # env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env

from pettingzoo.utils.conversions import parallel_wrapper_fn
env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

if __name__ == "__main__":

    e = env(render_mode="human")
    e.reset()
    # print(type(e))
    # while e.agents:
    #     actions = {agent: env.action_space(agent).sample() for agent in e.agents}
    #     e.render()
    #     observations, rewards, terminations, truncations, infos = e.step(actions)
    # e.close()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample() # this is where you would insert your policy

        env.step(action)
    env.close()