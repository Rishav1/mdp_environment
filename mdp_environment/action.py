from mdp_environment.exceptions import *

class Action:
	"""docstring for Action"""
	def __init__(self, id, kwargs):
		self.id = id
		for key, value in kwargs.items():
			setattr(self, key, value)
		if not hasattr(self, "name"):
			raise ActionNameNotProvided()

class ActionGenerator:
	"""docstring for ActionGenerator"""
	uid = -1
	unames = set()
	def __init__(self, *args):
		self.args = args

	def generate_action(self, **kwargs):
		if len(self.args) != len(kwargs.keys()):
			raise ActionParameterLengthMismatch()

		for (g_arg, i_arg) in zip(sorted(self.args), sorted(kwargs.keys())):
			if g_arg != i_arg:
				raise ActionParameterUndefined(i_arg)

		self.uid = self.uid + 1
		action =  Action(self.uid, kwargs)

		if action.name in self.unames:
			raise ActionNameNotUnique(action.name)

		self.unames.add(action.name)
		return action
		
