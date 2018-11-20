from mdp_environment.utils.exceptions import *

class State:
	"""docstring for State"""
	def __init__(self, id, kwargs ):
		self.id = id
		for key, value in kwargs.items():
			setattr(self, key, value)
		if not hasattr(self, "name"):
			raise StateNameNotProvided()


class StateGenerator:
	"""docstring for StateGenerator"""
	uid = -1
	def __init__(self, *args):
		self.args = args
		self.unames = set()

	def generate_state(self, **kwargs):
		if len(self.args) != len(kwargs.keys()):
			raise StateParameterLengthMismatch()

		for (g_arg, i_arg) in zip(sorted(self.args), sorted(kwargs.keys())):
			if g_arg != i_arg:
				raise StateParameterUndefined(i_arg)

		self.uid = self.uid + 1
		state = State(self.uid, kwargs)

		if state.name in self.unames:
			raise StateNameNotUnique(state.name)

		self.unames.add(state.name)
		return state

