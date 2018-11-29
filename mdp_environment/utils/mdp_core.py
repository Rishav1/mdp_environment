import numpy as np
from mdp_environment.utils.exceptions import *
import warnings
import networkx as nx
import os

class MDPModel:
	"""docstring for MDPModel"""
	def __init__(self, name, states = None, actions = None, transitions = None, init_states = None, final_states = None):
		self.name = name
		if states is None:
			self.states = {}
		if actions is None:
			self.actions = {}
		if transitions is None:
			self.transitions = {}
		if init_states is None:
			self.init_states = {}
		if final_states is None:
			self.final_states = {}
		self.finalized = False
		self.step=0
		self.final_step=-1
		self.initialized = False
		self.terminated = True
		self.visual_graph = nx.MultiDiGraph()

	def add_states(self, input_states):
		if self.finalized:
			raise MDPModelFinalized
		for state in input_states:
			if state.id in self.states.keys():
				raise StateAlreadyPresent({state.id: self.states[state.id].name })
			self.states[state.id] = state
		return self

	def get_states(self, state_ids):
		if type(state_ids) != list:
			if state_ids not in self.states.keys():
				raise StateNotPresent(state_ids)
			else:
				yield self.states[state_ids]
		for state_id in state_ids:
			if state_id not in self.states.keys():
				raise StateNotPresent(state_id)
			else:
				yield self.states[state_id]

	def add_actions(self, input_actions):
		if self.finalized:
			raise MDPModelFinalized
		for action in input_actions:
			if action.id in self.actions.keys():
				raise ActionAlreadyPresent({action.id: self.actions[action.id].name})
			self.actions[action.id] = action
		return self

	def get_actions(self, action_ids):
		if type(action_ids) != list:
			if action_ids not in self.actions.keys():
				raise ActionNotPresent(action_ids)
			else:
				yield self.actions[action_ids]
		for action_id in action_ids:
			if action_id not in self.actions.keys():
				raise ActionNotPresent(action_id)
			else:
				yield self.actions[action_id]

	def add_transition(self, state, action, p_transistion):
		if self.finalized:
			raise MDPModelFinalized
		if state.id not in self.states:
			raise StateNotPresent(state.id)
		if action.id not in self.actions:
			raise ActionNotPresent(action.id)
		for tstate in p_transistion.keys():
			if tstate.id not in self.states:
				raise StateNotPresent(tstate.id)
		try:
			np.testing.assert_almost_equal(np.sum(list(p_transistion.values())), 1.0)
		except AssertionError:
			raise ProbabilityError(p_transistion.values())
		if state.id in self.transitions:
			if action.id in self.transitions[state.id]:
				warnings.warn("Chaging transition probability at {0}.".format((state.id, action.id)))
		if state.id in self.transitions.keys():
			self.transitions[state.id][action.id] = p_transistion
		else:
			self.transitions[state.id] = {action.id: p_transistion}
		for tstate, prob in p_transistion.items():
			self.visual_graph.add_edge(state.name, tstate.name, weight=prob, label="P({0})={1}".format(action.name, prob))
		return self

	def add_init_states(self, init_states):
		if self.finalized:
			raise MDPModelFinalized
		for state in init_states.keys():
			if state.id not in self.states:
				raise StateNotPresent(state.id)
		try:
			np.testing.assert_almost_equal(np.sum(list(init_states.values())), 1.0)
		except AssertionError:
			raise ProbabilityError(init_states.values())
		self.init_states = init_states
		return self

	def add_final_states(self, final_states, final_step=-1):
		if self.finalized:
			raise MDPModelFinalized
		for state in final_states:
			if state.id not in self.states:
				raise StateNotPresent(state.id)
			self.final_states[state.id] = state
		self.final_step = final_step
		return self

	def finalize(self):
		if not self.init_states:
			raise InitStateNotSet
		self.finalized = True

	def initialize(self):
		if not self.finalized:
			raise MDPModelNotFinalized
		sample = np.random.multinomial(1, list(self.init_states.values()), size=1)
		index = np.where(sample[0]==1)[0][0]
		self.current_state = list(self.init_states.keys())[index]
		self.initialized = True
		self.terminated = False
		return self.current_state

	def transition(self, action):
		if not self.initialized:
			raise MDPModelNotInitialized
		if self.current_state.id not in self.transitions:
			raise InvalidAction((self.current_state, action))
		if action.id not in self.transitions[self.current_state.id]:
			raise InvalidAction((self.current_state, action))
		sample = np.random.multinomial(
			1,
			list(self.transitions[self.current_state.id][action.id].values()),
			1
		)
		index = np.where(sample[0]==1)[0][0]
		self.current_state = list(self.transitions[self.current_state.id][action.id].keys())[index]
		self.step += 1
		if (self.current_state.id not in self.transitions) or (self.current_state.id in self.final_states) or (self.step == self.final_step):
			self.terminated = True
			self.initialized = False
			self.step = 0

		return self.current_state

	def is_terminated(self):
		return self.terminated

	def visualize(self, path="/tmp"):
		if not self.finalized:
			raise MDPModelNotFinalized
		nx.drawing.nx_pydot.write_dot(self.visual_graph, '{0}/{1}.dot'.format(path, self.name))
		os.system("neato -Tps -Goverlap=scale {0}/{1}.dot -o {0}/{1}.ps; convert {0}/{1}.ps {0}/{1}.png; #rm {0}/{1}.dot {0}/{1}.ps".format(path, self.name))
