from mdp_environment.utils.exceptions import DefaultException
import pytest
import gym

def _run_lin_mdp(key):
    env = gym.make(key)
    env.reset()
    for _ in range(1000):
        env.render()
        _, _, over, _ = env.step(env.action_space.sample())
        if over:
            env.reset()

def test_lin_env_v0():
    try:
        _run_lin_mdp("mdp-v0")
    except DefaultException:
        pytest.fail("{0}: Unexpected error ..".format("mdp-v0"))

def test_lin_env_v1():
    try:
        _run_lin_mdp("mdp-v1")
    except DefaultException:
        pytest.fail("{0}: Unexpected error ..".format("mdp-v1"))