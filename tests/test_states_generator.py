from mdp_environment.utils.state import State, StateGenerator
from mdp_environment.utils.exceptions import *
import pytest

def test_argument_mismatch():
	g = StateGenerator('name', 'reward', 'probability')
	with pytest.raises(StateParameterLengthMismatch):
		g.generate_state(name="Test", reward=1)

def test_argument_undefined():
	g = StateGenerator('name', 'reward', 'probability')
	with pytest.raises(StateParameterUndefined):
		g.generate_state(name="Test", reward=1, prob=0.5)

def test_argument_has_name():
	g = StateGenerator('reward', 'probability', 'order')
	with pytest.raises(StateNameNotProvided):
		g.generate_state(reward=1, probability=0.5, order=0)
