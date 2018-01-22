import pytest
from mdp_environment.action import ActionGenerator
from mdp_environment.state import StateGenerator
from mdp_environment.mdp_core import MDPModel
from mdp_environment.exceptions import *

def test_state_duplicate_add():
	g = StateGenerator('name', 'reward')
	s0 = g.generate_state(name='s0', reward=1)
	s1 = g.generate_state(name='s1', reward=-1)
	m = MDPModel('Test')
	m.add_states([s0, s1])
	with pytest.raises(StateAlreadyPresent):
		m.add_states([s0])

def test_state_not_present():
	g = StateGenerator('name', 'reward')
	s0 = g.generate_state(name='s0', reward=1)
	s1 = g.generate_state(name='s1', reward=-1)
	s2 = g.generate_state(name='s2', reward=0)
	s3 = g.generate_state(name='s3', reward=0)
	m = MDPModel('Test')
	m.add_states([s0, s2])
	print(m.states)
	with pytest.raises(StateNotPresent):
		list(m.get_states([s1.id, s3.id]))

def test_action_duplicate_add():
	g = ActionGenerator('name', 'reward')
	a0 = g.generate_action(name='a0', reward=1)
	a1 = g.generate_action(name='a1', reward=-1)
	m = MDPModel('Test')
	m.add_actions([a0, a1])
	with pytest.raises(ActionAlreadyPresent):
		m.add_actions([a0])

def test_action_not_present():
	g = ActionGenerator('name', 'reward')
	a0 = g.generate_action(name='a0', reward=1)
	a1 = g.generate_action(name='a1', reward=-1)
	a2 = g.generate_action(name='a2', reward=0)
	a3 = g.generate_action(name='a3', reward=0)
	m = MDPModel('Test')
	m.add_actions([a0, a2])
	print(m.states)
	with pytest.raises(ActionNotPresent):
		list(m.get_actions([a1.id, a3.id]))

def test_faulty_transition_entry():
	s = StateGenerator('name', 'reward')
	s0 = s.generate_state(name='s0', reward=1)
	s1 = s.generate_state(name='s1', reward=-1)
	s2 = s.generate_state(name='s2', reward=0)
	s3 = s.generate_state(name='s3', reward=0)
	a = ActionGenerator('name', 'reward')
	a0 = a.generate_action(name='a0', reward=1)
	a1 = a.generate_action(name='a1', reward=-1)
	a2 = a.generate_action(name='a2', reward=-1)
	m = MDPModel('Test')
	m.add_states([s0, s1, s3])
	m.add_actions([a0, a1])
	with pytest.raises(StateNotPresent):
		m.add_transition(s2, a0, {s0.id: 0.5, s1.id: 0.5})
	with pytest.raises(ActionNotPresent):
		m.add_transition(s1, a2, {s0.id: 0.5, s1.id: 0.5})
	with pytest.raises(ProbabilityError):
		m.add_transition(s1, a0, {s0: 0.5, s1: 0.45})
	m.add_transition(s1, a0, {s0: 0.5, s1: 0.5})

def test_init_state_initialization():
	s = StateGenerator('name', 'reward')
	s0 = s.generate_state(name='s0', reward=1)
	s1 = s.generate_state(name='s1', reward=-1)
	p = {s0: 0.5, s1: 0.45}
	m = MDPModel('Test')
	m.add_states([s0])
	with pytest.raises(StateNotPresent):
		m.add_init_states(p)
	m.add_states([s1])
	with pytest.raises(ProbabilityError):
		m.add_init_states(p)
	p[s1] = 0.5
	m.add_init_states(p)

def test_finalize():
	m = MDPModel('Test')
	with pytest.raises(InitStateNotSet):
		m.finalize()
	s = StateGenerator('name', 'reward')
	s0 = s.generate_state(name='s0', reward=1)
	s1 = s.generate_state(name='s1', reward=1)
	a = ActionGenerator('name', 'reward')
	a0 = a.generate_action(name='a0', reward=1)
	a1 = a.generate_action(name='a1', reward=1)
	p = {s0: 0.5, s1: 0.5}
	m.add_states([s0])
	m.add_actions([a0])
	m.add_init_states({s0: 1})
	m.finalize()
	with pytest.raises(MDPModelFinalized):
		m.add_states([s1])
		m.add_actions([a1])
		m.add_init_states(p)
		m.add_transition(s0, a0, p)

def test_initialize():
	m = MDPModel('Test')
	with pytest.raises(InitStateNotSet):
		m.finalize()
	s = StateGenerator('name', 'reward')
	s0 = s.generate_state(name='s0', reward=1)
	s1 = s.generate_state(name='s1', reward=1)
	a = ActionGenerator('name', 'reward')
	a0 = a.generate_action(name='a0', reward=1)
	a1 = a.generate_action(name='a1', reward=1)
	p = {s0: 0.5, s1: 0.5}
	m.add_states([s0])
	m.add_actions([a0])
	m.add_init_states({s0: 1})
	with pytest.raises(MDPModelNotFinalized):
		m.initialize()
	m.finalize()
	m.initialize()

def test_transitions():
	s = StateGenerator('name', 'reward')
	s0 = s.generate_state(name='s0', reward=1)
	s1 = s.generate_state(name='s1', reward=-1)
	s2 = s.generate_state(name='s2', reward=0)
	s3 = s.generate_state(name='s3', reward=0)
	s4 = s.generate_state(name='s4', reward=0)
	a = ActionGenerator('name', 'reward')
	a0 = a.generate_action(name='a0', reward=1)
	a1 = a.generate_action(name='a1', reward=-1)
	a2 = a.generate_action(name='a2', reward=-1)
	m = MDPModel('Test')
	m.add_states([s0, s1, s2, s3, s4]).add_actions([a0, a1, a2]).add_init_states({s0: 1}) \
	.add_transition(s0, a0, {s1: 1}).add_transition(s1, a0, {s2: 1}) \
	.add_transition(s2, a1, {s3: 1}).add_transition(s3, a2, {s4: 1}).finalize()
	
	assert m.initialize() == s0
	assert m.transition(a0) == s1
	with pytest.raises(InvalidAction):
		m.transition(a1)
	assert m.transition(a0) == s2
	assert m.transition(a1) == s3
	assert m.transition(a2) == s4
	assert m.is_terminated() == True
