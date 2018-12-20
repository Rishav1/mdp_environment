class DefaultException(Exception):
    """docstring for DefaultException(Exception)"""

    def __init__(self, m):
        self.m = m

    def __str__(self):
        return self.m


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


class ProbabilityError(Exception):
    """docstring for ProbabilityError"""

    def __init__(self, dErrorArguments):
        Exception.__init__(self, "Sum of probabilities in {0} is not unity".format(dErrorArguments))
        self.dErrorArguments = dErrorArguments


class MDPModelFinalized(Exception):
    """docstring for MDPModelFinalized"""

    def __init__(self):
        Exception.__init__(self, "MDP Model is already finalized.")


class MDPModelNotFinalized(Exception):
    """docstring for MDPModelNotFinalized"""

    def __init__(self):
        Exception.__init__(self, "MDP Model hasn't been finalized.")


class MDPModelNotInitialized(Exception):
    """docstring for MDPModelNotInitialized"""

    def __init__(self):
        Exception.__init__(self, "MDP Model hasn't been initialized.")


class InitStateNotSet(Exception):
    """docstring for InitStateNotSet"""

    def __init__(self):
        Exception.__init__(self, "Start State not set.")


class InvalidAction(Exception):
    """docstring for InvalidAction"""

    def __init__(self, dErrorArguments):
        Exception.__init__(self,
                           "Action {0} at state {0} is not feasible.".format(dErrorArguments[1], dErrorArguments[0]))
        self.dErrorArguments = dErrorArguments
