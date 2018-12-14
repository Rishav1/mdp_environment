import gym
from gym import spaces
from gym.utils import seeding
from mdp_environment.utils.state import StateGenerator
from mdp_environment.utils.action import ActionGenerator
from mdp_environment.utils.mdp_core import MDPModel
import numpy as np
from scipy.stats import binom

class MdpEnvLinStatic(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, size=15, prob=0.2, path="/tmp"):
    self.env = self._create_mdp(path, prob, size)
    self.observation_space = spaces.Discrete(size+1)
    self.action_space=spaces.Discrete(2)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    np.random.seed(seed)
    return [seed]

  def _create_mdp(self, path, prob, size):
    def _fill_reward(x):
      if (x == 0):
        return 0.1
      elif (x == size - 1):
        return 1
      else:
        return 0

    stateGenerator = StateGenerator('name', 'reward')
    actionGenerator = ActionGenerator('name')
    states = [stateGenerator.generate_state(name="s" + str(x), reward=_fill_reward(x)) for x in range(size)]
    actions = [actionGenerator.generate_action(name=name) for name in ['LEFT', 'RIGHT']]
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
    # mdp.visualize(path)
    return mdp

  def step(self, action_id):
    obs = self.env.transition(next(self.env.get_actions(action_id)))
    reward = obs.reward
    over = self.env.is_terminated()
    self.rewards.append(reward)
    self.states.append(obs)
    info = {"rewards": self.rewards, "states": self.states}
    return obs.id, reward, over, info

  def reset(self):
    obs = self.env.initialize()
    self.rewards = []
    self.states = [obs]
    return obs.id


  def render(self, mode='human', close=False):
    ...

class MdpEnvLinVariable(MdpEnvLinStatic):

  def __init__(self, size=15, prob=0.2, path="/tmp"):
    MdpEnvLinStatic.__init__(self, size, prob, path)
    self.env = self._create_mdp(path, prob, size)

  def _create_mdp(self, path, prob, size):
    def _fill_reward(x):
      if (x == 0):
        return 0.1
      elif (x == size - 1):
        return 1
      else:
        return 0

    stateGenerator = StateGenerator('name', 'reward')
    actionGenerator = ActionGenerator('name')
    states = [stateGenerator.generate_state(name="s" + str(x), reward=_fill_reward(x)) for x in range(size)]
    actions = [actionGenerator.generate_action(name=name) for name in ['LEFT', 'RIGHT']]

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
    # mdp.visualize(path)
    return mdp

class MdpEnvPlanarStatic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=10, prob=0.2, path="/tmp"):
      self.env = self._create_mdp(path, prob, size)
      self.observation_space = spaces.Discrete(size**2)
      self.action_space = spaces.Discrete(4)

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      np.random.seed(seed)
      return [seed]

    def _create_mdp(self, path, prob, size):
      def _fill_reward(x):
        if (x == (0,0)):
          return 0.1
        elif (x == (size - 1, size - 1)):
          return 1
        else:
          return 0

      stateGenerator = StateGenerator('name', 'reward')
      actionGenerator = ActionGenerator('name')
      states = [[stateGenerator.generate_state(name="s" + str(x) + "-" + str(y), reward=_fill_reward((x,y))) for x in range(size)] for y in range(size)]
      actions = [actionGenerator.generate_action(name=name) for name in ['LEFT', 'RIGHT', 'DOWN', 'UP']]
      mdp = MDPModel('MdpEnvPlanarStatic') \
        .add_states([item for sublist in states for item in sublist]) \
        .add_actions(actions) \
        .add_init_states({states[1 + np.random.binomial(size - 3, prob)][1 + np.random.binomial(size - 3, prob)]: 1}) \
        .add_final_states([states[x][x] for x in [0, size - 1]], 1000)

      for j in range(size):
        mdp \
          .add_transition(states[j][0], actions[0], {states[j][0]: 1}) \
          .add_transition(states[j][size-1], actions[1], {states[j][size-1]: 1})
        for i in range(-1, size-1):
          if i==-1:
            mdp.add_transition(states[j][i + 1], actions[1], {states[j][i + 2]: 1})
          elif i==size-2:
            mdp.add_transition(states[j][i + 1], actions[0], {states[j][i]: 1})
          else:
            mdp \
              .add_transition(states[j][i + 1], actions[0], {states[j][i]: 1}) \
              .add_transition(states[j][i + 1], actions[1], {states[j][i + 2]: 1})

      for j in range(size):
        mdp \
          .add_transition(states[0][j], actions[2], {states[0][j]: 1}) \
          .add_transition(states[size-1][j], actions[3], {states[size-1][j]: 1})
        for i in range(-1, size-1):
          if i==-1:
            mdp.add_transition(states[i + 1][j], actions[3], {states[i + 2][j]: 1})
          elif i==size-2:
            mdp.add_transition(states[i + 1][j], actions[2], {states[i][j]: 1})
          else:
            mdp \
              .add_transition(states[i + 1][j], actions[2], {states[i][j]: 1}) \
              .add_transition(states[i + 1][j], actions[3], {states[i + 2][j]: 1})

      # Visualize the MDP
      mdp.finalize()
      # mdp.visualize(path)
      return mdp

    def step(self, action_id):
      obs = self.env.transition(next(self.env.get_actions(action_id)))
      reward = obs.reward
      over = self.env.is_terminated()
      self.rewards.append(reward)
      self.states.append(obs)
      info = {"rewards": self.rewards , "states": self.states}
      return obs.id, reward, over, info

    def reset(self):
      obs = self.env.initialize()
      self.rewards = []
      self.states = [obs]
      return obs.id

    def render(self, mode='human', close=False):
      ...

class MdpEnvPlanarVariable(MdpEnvPlanarStatic):

  def __init__(self, size=10, prob=0.2, path="/tmp"):
    MdpEnvPlanarStatic.__init__(self, size, prob, path)
    self.env = self._create_mdp(path, prob, size)

  def _create_mdp(self, path, prob, size):
    def _fill_reward(x):
      if (x == (0, 0)):
        return 0.1
      elif (x == (size - 1, size - 1)):
        return 1
      else:
        return 0

    stateGenerator = StateGenerator('name', 'reward')
    actionGenerator = ActionGenerator('name')
    states = [
      [stateGenerator.generate_state(name="s" + str(x) + "-" + str(y), reward=_fill_reward((x,y))) for x in range(size)] for
      y in range(size)]
    actions = [actionGenerator.generate_action(name=name) for name in ['LEFT', 'RIGHT', 'DOWN', 'UP']]

    init_states = dict(zip([item for sublist in states for item in sublist], [binom.pmf(i, size - 1, prob) * binom.pmf(j, size - 1, prob) for i in range(size) for j in range(size)]))

    mdp = MDPModel('MdpEnvPlanarStatic') \
      .add_states([item for sublist in states for item in sublist]) \
      .add_actions(actions) \
      .add_init_states(init_states) \
      .add_final_states([states[x][x] for x in [0, size - 1]],1000)

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
    # mdp.visualize(path)
    return mdp

  class MdpEnvPlanarStatic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=10, prob=0.2, path="/tmp"):
      self.env = self._create_mdp(path, prob, size)
      self.observation_space = spaces.Discrete(size ** 2)
      self.action_space = spaces.Discrete(4)

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      np.random.seed(seed)
      return [seed]

    def _create_mdp(self, path, prob, size):
      def _fill_reward(x):
        if (x == (0, 0)):
          return 0.1
        elif (x == (size - 1, size - 1)):
          return 1
        else:
          return 0

      stateGenerator = StateGenerator('name', 'reward')
      actionGenerator = ActionGenerator('name')
      states = [[stateGenerator.generate_state(name="s" + str(x) + "-" + str(y), reward=_fill_reward((x, y))) for x in
                 range(size)] for y in range(size)]
      actions = [actionGenerator.generate_action(name=name) for name in ['LEFT', 'RIGHT', 'DOWN', 'UP']]
      mdp = MDPModel('MdpEnvPlanarStatic') \
        .add_states([item for sublist in states for item in sublist]) \
        .add_actions(actions) \
        .add_init_states({states[1 + np.random.binomial(size - 3, prob)][1 + np.random.binomial(size - 3, prob)]: 1}) \
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
      # mdp.visualize(path)
      return mdp

    def step(self, action_id):
      obs = self.env.transition(next(self.env.get_actions(action_id)))
      reward = obs.reward
      over = self.env.is_terminated()
      self.rewards.append(reward)
      self.states.append(obs)
      info = {"rewards": self.rewards, "states": self.states}
      return obs.id, reward, over, info

    def reset(self):
      obs = self.env.initialize()
      self.rewards = []
      self.states = [obs]
      return obs.id

    def render(self, mode='human', close=False):
      ...