import gym
import pytest

from mdp_environment.utils.exceptions import DefaultException


def _run_lin_mdp(key):
    env = gym.make(key)
    env.reset()
    for i in range(1000):
        env.render(close=False)
        _, reward, over, _ = env.step(env.action_space.sample())
        if over:?!?jedi=0, ?!? (*_*value*_*, ..., sep, end, file, flush) ?!?jedi?!?
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


def test_lin_env_v2():
    try:
        _run_lin_mdp("mdp-v2")
    except DefaultException:
        pytest.fail("{0}: Unexpected error ..".format("mdp-v2"))


def test_lin_env_v3():
    try:
        _run_lin_mdp("mdp-v3")
    except DefaultException:
        pytest.fail("{0}: Unexpected error ..".format("mdp-v3"))


def test_lin_env_v4():
    try:
        _run_lin_mdp("mdp-v4")
    except DefaultException:
        pytest.fail("{0}: Unexpected error ..".format("mdp-v4"))
