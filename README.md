# A generic MDP gym environment.

I am building this environment primarily for my Reinforcement Learning research. Purpose of this python module is to enable creation and simulation of Markov Decision Processes.

The environment is accessible through the OpenAI gym wrapper. An example to use it as follows.

```python
import gym
import mdp_environment

env = gym.make("mdp-v0")
env.reset()
for _ in range(1000):
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()

```

The "mdp-v0" is a linear chain of 11 states, with initial state selected via binomial distribution(size=10, prob=5). The final states are at s0, s5, s10. Rewards are 0 for any transition except for terninating transitions. Terminating transition to state s5 is 0.1, while that to s0 or s10 is 1.

![MDP transtion](https://i.imgur.com/kSYCUEx.png)