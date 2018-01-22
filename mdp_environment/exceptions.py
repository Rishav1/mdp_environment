
class StateAlreadyPresent(Exception):
	"""docstring for StateAlreadyPresent(Exception)"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "State ID {0} already exists".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments

class StateNotPresent(Exception):
	"""docstring for StateNotPresent(Exception)"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "State ID {0} dosent exist".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments

class StateParameterLengthMismatch(Exception):
	"""docstring for StateNotPresent(Exception)"""
	def __init__(self):
		Exception.__init__(self, "State parameters count dosent match generator's count")

class StateParameterUndefined(Exception):
	"""docstring for StateNotPresent(Exception)"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "State Parameter {0} dosent exist".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments

class StateNameNotProvided(Exception):
	"""docstring for StateNotPresent(Exception)"""
	def __init__(self):
		Exception.__init__(self, "State wasn't provided a name.")

class StateNameNotUnique(Exception):
	"""docstring for StateNameNotUnique"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "State with name {0} has already been defined.".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments
		


class ActionAlreadyPresent(Exception):
	"""docstring for ActionAlreadyPresent(Exception)"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "Action ID {0} already exists".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments

class ActionNotPresent(Exception):
	"""docstring for ActionNotPresent(Exception)"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "Action ID {0} dosent exist".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments

class ActionParameterLengthMismatch(Exception):
	"""docstring for ActionNotPresent(Exception)"""
	def __init__(self):
		Exception.__init__(self, "Action parameters count dosent match generator's count")

class ActionParameterUndefined(Exception):
	"""docstring for ActionNotPresent(Exception)"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "Action Parameter {0} dosent exist".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments

class ActionNameNotProvided(Exception):
	"""docstring for ActionNotPresent(Exception)"""
	def __init__(self):
		Exception.__init__(self, "Action wasn't provided a name.")

class ActionNameNotUnique(Exception):
	"""docstring for ActionNameNotUnique"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "Action with name {0} has already been defined.".format(dErrorArguments))
		self.dErrorArguments = dErrorArguments


class TransitionProbabilityError:
	"""docstring for TransitionProbabilityError"""
	def __init__(self, dErrorArguments):
		Exception.__init__(self, "Transition Probability from state {0} on action {1} \
			does not sum to unity".format(dErrorArguments[0], dErrorArguments[1]))
		self.dErrorArguments = dErrorArguments
		