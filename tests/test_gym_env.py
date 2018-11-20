from mdp_environment.utils.exceptions import DefaultException
import pytest
import gym

def _run_lin_mdp():
    env = gym.make("mdp-v0")
    env.reset()
    for _ in range(1000):
        env.render()
        _, _, over, _ = env.step(env.action_space.sample())
        if over:
            env.reset()

def test_lin_env():
    try:
        _run_lin_mdp()
    except DefaultException:
        pytest.fail("Unexpected error ..")