from mdp_environment.utils.action import ActionGenerator
from mdp_environment.utils.exceptions import *
import pytest

def test_argument_mismatch():
	g = ActionGenerator('name', 'reward', 'probability')
	with pytest.raises(ActionParameterLengthMismatch):
		g.generate_action(name="Test", reward=1)

def test_argument_undefined():
	g = ActionGenerator('name', 'reward', 'probability')
	with pytest.raises(ActionParameterUndefined):
		g.generate_action(name="Test", reward=1, prob=0.5)

def test_argument_has_name():
	g = ActionGenerator('reward', 'probability', 'order')
	with pytest.raises(ActionNameNotProvided):
		g.generate_action(reward=1, probability=0.5, order=0)
