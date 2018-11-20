import gym
from gym import spaces
from mdp_environment.utils.state import StateGenerator
from mdp_environment.utils.action import ActionGenerator
from mdp_environment.utils.mdp_core import MDPModel
import numpy as np

class MdpEnvLin(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, seed=-1, size=10, prob=0.2, path="/tmp"):
    self.env = self._create_mdp(path, prob, size, seed)
    self.observation_space = spaces.Discrete(size+1)
    self.action_space=spaces.Discrete(2)

  def _create_mdp(self, path, prob, size, seed):
    def _fill_reward(x):
      if (x == 0):
        return 0.1
      elif (x == size - 1):
        return 1
      else:
        return 0

    if(seed!=-1):
      np.random.seed(seed)

    stateGenerator = StateGenerator('name', 'reward')
    actionGenerator = ActionGenerator('name')
    states = [stateGenerator.generate_state(name="s" + str(x), reward=_fill_reward(x)) for x in range(size)]
    actions = [actionGenerator.generate_action(name=name) for name in ['LEFT', 'RIGHT']]
    mdp = MDPModel('MdpEnvLin') \
      .add_states(states) \
      .add_actions(actions) \
      .add_init_states({states[np.random.binomial(size - 1, prob)]: 1}) \
      .add_final_states([states[x] for x in [0, size - 1]])

    for i in range(size - 2):
      mdp \
        .add_transition(states[i + 1], actions[0], {states[i]: 1}) \
        .add_transition(states[i + 1], actions[1], {states[i + 2]: 1})
    # Visualize the MDP
    mdp.finalize()
    mdp.visualize(path)
    return mdp

  def step(self, action_id):
    obs = self.env.transition(next(self.env.get_actions(action_id)))
    reward = obs.reward
    over = self.env.is_terminated()
    info = {"rewards": self.rewards, "states": self.states}
    return obs, reward, over, info

  def reset(self):
    obs = self.env.initialize()
    self.rewards = []
    self.states = [obs]
    return obs


  def render(self, mode='human', close=False):
    ...