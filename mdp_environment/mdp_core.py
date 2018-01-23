import numpy as np
from mdp_environment.exceptions import *
import warnings
import networkx as nx 
import matplotlib.pyplot as plt 
import os

class MDPModel:
	"""docstring for MDPModel"""
	def __init__(self, name, states = None, actions = None, transitions = None, init_states = None):
		self.name = name
		if states is None:
			self.states = {}
		if actions is None:
			self.actions = {}
		if transitions is None:
			self.transitions = {}
		if init_states is None:
			self.init_states = {}
		self.finalized = False
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
		if type(state_ids) == int:
			if state_ids not in self.states.keys():
				raise StateNotPresent(state_ids)
			else:
				return self.states[state_ids]
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
		if type(action_ids) == int:
			if action_ids not in self.actions.keys():
				return self.actions[action_ids]
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
				warnings.warn("Chaging transition probability at {0}.".format((state.id, actions.id)))
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

		if self.current_state.id not in self.transitions:
			self.terminated = True
			self.initialized = False

		return self.current_state

	def is_terminated(self):
		return self.terminated

	def visualize(self):
		if not self.finalized:
			raise MDPModelNotFinalized
		nx.drawing.nx_pydot.write_dot(self.visual_graph, '{0}.dot'.format(self.name))
		os.system("neato -Tps -Goverlap=scale {0}.dot -o {0}.ps; convert {0}.ps {0}.png; rm {0}.dot {0}.ps".format(self.name))

