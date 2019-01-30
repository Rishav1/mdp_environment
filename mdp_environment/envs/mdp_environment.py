import pyglet
from matplotlib import pyplot
import io
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy.stats import binom
from mdp_environment.utils.action import ActionGenerator
from mdp_environment.utils.mdp_core import MDPModel
from mdp_environment.utils.state import StateGenerator


class MdpEnvLinStatic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=15, prob=0.2, path="/tmp"):
        self.viewer = None
        self.size = size
        self.env = self._create_mdp(path, prob, size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(size,), dtype=np.int32)
        self.action_space = spaces.Discrete(2)
        self.rewards = None
        self.states = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _create_mdp(self, path, prob, size):
        self.path = path
        def _fill_reward(x):
            if x == 0:
                return 0.1
            elif x == size - 1:
                return 1
            else:
                return 0

        state_generator = StateGenerator('name', 'reward')
        action_generator = ActionGenerator('name')
        states = [state_generator.generate_state(name="s" + str(x), reward=_fill_reward(x)) for x in range(size)]
        actions = [action_generator.generate_action(name=name) for name in ['LEFT', 'RIGHT']]
        mdp = MDPModel('MdpEnvLinStatic') \
            .add_states(states) \
            .add_actions(actions) \
            .add_init_states({states[1 + np.random.binomial(size - 3, prob)]: 1}) \
            .add_final_states([states[x] for x in [0, size - 1]], 100)

        for i in range(size - 2):
            mdp \
                .add_transition(states[i + 1], actions[0], {states[i]: 1}) \
                .add_transition(states[i + 1], actions[1], {states[i + 2]: 1})
        # Visualize the MDP
        mdp.finalize()
        # mdp.visualize(file="{0}/{1}".format(path, self.__class__.__name__))
        return mdp

    def step(self, action_id):
        obs = self.env.transition(next(self.env.get_actions(action_id)))
        reward = obs.reward
        over = self.env.is_terminated()
        self.rewards.append(reward)
        self.states.append(obs)
        info = {"rewards": self.rewards, "states": self.states}
        return np.eye(self.size)[obs.id], reward, over, info

    def reset(self):
        obs = self.env.initialize()
        self.rewards = []
        self.states = [obs]
        return np.eye(self.size)[obs.id]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self.env.visualize(file="{0}/{1}".format(self.path, self.__class__.__name__))

        if mode == 'rgb_array':
            return pyplot.imread(io.BytesIO(img))
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            image = pyglet.image.load(filename="{0}/{1}".format(self.path, self.__class__.__name__))
            self.viewer.imgshow(image)
            return self.viewer.isopen
        elif mode == 'png':
            return img


class MdpEnvLinVariable(MdpEnvLinStatic):
    def __init__(self, size=15, prob=0.2, path="/tmp"):
        MdpEnvLinStatic.__init__(self, size, prob, path)
        self.env = self._create_mdp(path, prob, size)

    def _create_mdp(self, path, prob, size):
        def _fill_reward(x):
            if x == 0:
                return 0.1
            elif x == size - 1:
                return 1
            else:
                return 0

        state_generator = StateGenerator('name', 'reward')
        action_generator = ActionGenerator('name')
        states = [state_generator.generate_state(name="s" + str(x), reward=_fill_reward(x)) for x in range(size)]
        actions = [action_generator.generate_action(name=name) for name in ['LEFT', 'RIGHT']]

        # Initializing the states of the mdp with binomial distribution
        init_states = dict(zip(states, [0] + [binom.pmf(i, size - 3, prob) for i in range(size - 2)] + [0]))

        mdp = MDPModel('MdpEnvLinVariable') \
            .add_states(states) \
            .add_actions(actions) \
            .add_init_states(init_states) \
            .add_final_states([states[x] for x in [0, size - 1]], 100)

        for i in range(size - 2):
            mdp \
                .add_transition(states[i + 1], actions[0], {states[i]: 1}) \
                .add_transition(states[i + 1], actions[1], {states[i + 2]: 1})
        # Visualize the MDP
        mdp.finalize()
        # mdp.visualize(file="{0}/{1}".format(path, self.__class__.__name__))
        return mdp


class MdpEnvPlanarStatic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=15, prob=0.2, path="/tmp"):
        self.viewer = None
        self.path = path
        self.env = self._create_mdp(path, prob, size)
        self.size = size
        self.observation_space = spaces.Box(low=0, high=1, shape=(size,size),dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.rewards = None
        self.states = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _create_mdp(self, path, prob, size):
        def _fill_reward(x):
            if x == (0, 0):
                return 0.1
            elif x == (size - 1, size - 1):
                return 1
            else:
                return 0

        state_generator = StateGenerator('name', 'reward')
        action_generator = ActionGenerator('name')
        states = [
            [state_generator.generate_state(name="s" + str(x) + "-" + str(y), reward=_fill_reward((x, y))) for x in
             range(size)] for y in range(size)]
        actions = [action_generator.generate_action(name=name) for name in ['LEFT', 'RIGHT', 'DOWN', 'UP']]
        mdp = MDPModel('MdpEnvPlanarStatic') \
            .add_states([item for sublist in states for item in sublist]) \
            .add_actions(actions) \
            .add_init_states(
            {states[1 + np.random.binomial(size - 3, prob)][1 + np.random.binomial(size - 3, prob)]: 1}) \
            .add_final_states([states[x][x] for x in [0, size - 1]], 1000)

        for j in range(size):
            mdp \
                .add_transition(states[j][0], actions[0], {states[j][0]: 1}) \
                .add_transition(states[j][size - 1], actions[1], {states[j][size - 1]: 1})
            for i in range(-1, size - 1):
                if i == -1:
                    mdp.add_transition(states[j][i + 1], actions[1], {states[j][i + 2]: 1})
                elif i == size - 2:
                    mdp.add_transition(states[j][i + 1], actions[0], {states[j][i]: 1})
                else:
                    mdp \
                        .add_transition(states[j][i + 1], actions[0], {states[j][i]: 1}) \
                        .add_transition(states[j][i + 1], actions[1], {states[j][i + 2]: 1})

        for j in range(size):
            mdp \
                .add_transition(states[0][j], actions[2], {states[0][j]: 1}) \
                .add_transition(states[size - 1][j], actions[3], {states[size - 1][j]: 1})
            for i in range(-1, size - 1):
                if i == -1:
                    mdp.add_transition(states[i + 1][j], actions[3], {states[i + 2][j]: 1})
                elif i == size - 2:
                    mdp.add_transition(states[i + 1][j], actions[2], {states[i][j]: 1})
                else:
                    mdp \
                        .add_transition(states[i + 1][j], actions[2], {states[i][j]: 1}) \
                        .add_transition(states[i + 1][j], actions[3], {states[i + 2][j]: 1})

        # Visualize the MDP
        mdp.finalize()
        # mdp.visualize(file="{0}/{1}".format(path, self.__class__.__name__))
        return mdp

    def step(self, action_id):
        obs = self.env.transition(next(self.env.get_actions(action_id)))
        observation = np.zeros((self.size, self.size))
        observation.__setitem__(self.get_obs_index(obs.id), 1)
        reward = obs.reward
        over = self.env.is_terminated()
        self.rewards.append(reward)
        self.states.append(obs)
        info = {"rewards": self.rewards, "states": self.states}
        return observation, reward, over, info

    def get_obs_index(self, id):
        return (int(id % self.size), int(id / self.size))

    def reset(self):
        obs = self.env.initialize()
        self.rewards = []
        self.states = [obs]
        observation = np.zeros((self.size, self.size))
        observation.__setitem__(self.get_obs_index(obs.id), 1)
        return observation

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self.env.visualize(file="{0}/{1}".format(self.path, self.__class__.__name__))

        if mode == 'rgb_array':
            return pyplot.imread(io.BytesIO(img))
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            image = pyglet.image.load(filename="{0}/{1}".format(self.path, self.__class__.__name__))
            self.viewer.imgshow(image)
            return self.viewer.isopen
        elif mode == 'png':
            return img


class MdpEnvPlanarVariable(MdpEnvPlanarStatic):
    def __init__(self, size=15, prob=0.2, path="/tmp"):
        MdpEnvPlanarStatic.__init__(self, size, prob, path)
        self.env = self._create_mdp(path, prob, size)

    def _create_mdp(self, path, prob, size):
        def _fill_reward(x):
            if x == (0, 0):
                return 0.1
            elif x == (size - 1, size - 1):
                return 1
            else:
                return 0

        state_generator = StateGenerator('name', 'reward')
        action_generator = ActionGenerator('name')
        states = [
            [state_generator.generate_state(name="s" + str(x) + "-" + str(y), reward=_fill_reward((x, y))) for x in
             range(size)] for
            y in range(size)]
        actions = [action_generator.generate_action(name=name) for name in ['LEFT', 'RIGHT', 'DOWN', 'UP']]

        init_states = dict(zip([item for sublist in states for item in sublist],
                               [binom.pmf(i, size - 1, prob) * binom.pmf(j, size - 1, prob) for i in range(size) for j
                                in range(size)]))

        mdp = MDPModel('MdpEnvPlanarStatic') \
            .add_states([item for sublist in states for item in sublist]) \
            .add_actions(actions) \
            .add_init_states(init_states) \
            .add_final_states([states[x][x] for x in [0, size - 1]], 1000)

        for j in range(size):
            mdp \
                .add_transition(states[j][0], actions[0], {states[j][0]: 1}) \
                .add_transition(states[j][size - 1], actions[1], {states[j][size - 1]: 1})
            for i in range(-1, size - 1):
                if i == -1:
                    mdp.add_transition(states[j][i + 1], actions[1], {states[j][i + 2]: 1})
                elif i == size - 2:
                    mdp.add_transition(states[j][i + 1], actions[0], {states[j][i]: 1})
                else:
                    mdp \
                        .add_transition(states[j][i + 1], actions[0], {states[j][i]: 1}) \
                        .add_transition(states[j][i + 1], actions[1], {states[j][i + 2]: 1})

        for j in range(size):
            mdp \
                .add_transition(states[0][j], actions[2], {states[0][j]: 1}) \
                .add_transition(states[size - 1][j], actions[3], {states[size - 1][j]: 1})
            for i in range(-1, size - 1):
                if i == -1:
                    mdp.add_transition(states[i + 1][j], actions[3], {states[i + 2][j]: 1})
                elif i == size - 2:
                    mdp.add_transition(states[i + 1][j], actions[2], {states[i][j]: 1})
                else:
                    mdp \
                        .add_transition(states[i + 1][j], actions[2], {states[i][j]: 1}) \
                        .add_transition(states[i + 1][j], actions[3], {states[i + 2][j]: 1})

        # Visualize the MDP
        mdp.finalize()
        # mdp.visualize(file="{0}/{1}".format(path, self.__class__.__name__))
        return mdp

class MdpEnvLinCorridor(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=25, dim=20, path="/tmp"):
        self.viewer = None
        self.size = size
        self.env = self._create_mdp(path, size, dim)
        self.observation_space = spaces.Box(low=0, high=1, shape=(size,), dtype=np.int32)
        self.action_space = spaces.Discrete(dim)
        self.rewards = None
        self.states = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _create_mdp(self, path, size, dim):
        self.path = path
        def _fill_reward(x):
            if x == size - 1:
                return 1
            else:
                return 0

        state_generator = StateGenerator('name', 'reward')
        action_generator = ActionGenerator('name')
        states = [state_generator.generate_state(name="s" + str(x), reward=_fill_reward(x)) for x in range(size)]
        actions = [action_generator.generate_action(name="A{}".format(name)) for name in range(dim)]
        mdp = MDPModel('MdpEnvLinStatic') \
            .add_states(states) \
            .add_actions(actions) \
            .add_init_states({states[0]: 1}) \
            .add_final_states([states[size - 1]], 1000)

        for i in range(size - 2):
            actions_r = np.random.choice(dim, 2, replace=False)
            for action in range(dim):
                if action not in actions_r:
                    mdp.add_transition(states[i + 1], actions[action], {states[i + 1]: 1})
                elif action == actions_r[0]:
                    mdp.add_transition(states[i + 1], actions[actions_r[0]], {states[i]: 1})
                elif action == actions_r[1]:
                    mdp.add_transition(states[i + 1], actions[actions_r[1]], {states[i + 2]: 1})

        actions_r = np.random.choice(dim, 1, replace=False)
        for action in range(dim):
            if action not in actions_r:
                mdp.add_transition(states[0], actions[action], {states[0]: 1})
            else:
                mdp.add_transition(states[0], actions[action], {states[1]: 1})

        # Visualize the MDP
        mdp.finalize()
        # mdp.visualize(file="{0}/{1}".format(path, self.__class__.__name__))
        return mdp

    def step(self, action_id):
        obs = self.env.transition(next(self.env.get_actions(action_id)))
        reward = obs.reward
        over = self.env.is_terminated()
        self.rewards.append(reward)
        self.states.append(obs)
        info = {"rewards": self.rewards, "states": self.states}
        return np.eye(self.size)[obs.id], reward, over, info

    def reset(self):
        obs = self.env.initialize()
        self.rewards = []
        self.states = [obs]
        return np.eye(self.size)[obs.id]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.env.visualize(file="{0}/{1}".format(self.path, self.__class__.__name__))
        if mode == 'rgb_array':
            return pyplot.imread(io.BytesIO(img))
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            image = pyglet.image.load(filename="{0}/{1}".format(self.path, self.__class__.__name__))
            self.viewer.imgshow(image)
            return self.viewer.isopen
        elif mode == 'png':
            return img
