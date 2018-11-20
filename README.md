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

There are two custom MDP environments with the following details.

- **mdp-v0**: 
  - **S**: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{S_0,&space;S_1,...,&space;S_{10}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\{S_0,&space;S_1,...,&space;S_{9}\}" title="\{S_0, S_1,..., S_{9}\}" /></a>
  
  - **A**: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{LEFT,&space;RIGHT\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\{LEFT,&space;RIGHT\}" title="\{LEFT, RIGHT\}" /></a>
  
  - **T**: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;T(S_i,LEFT)&space;=&space;S_{i-1}&space;\quad&space;\&&space;\quad&space;T(S_i,RIGHT)&space;=&space;S_{i&plus;1}&space;\&space;\forall&space;i&space;\in&space;\{1,&space;2,&space;..,&space;8\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;T(S_i,LEFT)&space;=&space;S_{i-1}&space;\quad&space;\&&space;\quad&space;T(S_i,RIGHT)&space;=&space;S_{i&plus;1}&space;\&space;\forall&space;i&space;\in&space;\{1,&space;2,&space;..,&space;8\}" title="T(S_i,LEFT) = S_{i-1} \quad \& \quad T(S_i,RIGHT) = S_{i+1} \ \forall i \in \{1, 2, .., 8\}" /></a>
  
  - **R**: <a href="https://www.codecogs.com/eqnedit.php?latex=R(S_1,&space;LEFT)&space;=&space;0.1,\quad&space;R(S_8,&space;RIGHT)&space;=&space;1&space;\quad&space;\&&space;\quad&space;R(S,&space;\cdot)&space;=&space;0&space;\&space;\forall&space;S&space;\neq&space;S_1,&space;S_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(S_1,&space;LEFT)&space;=&space;0.1,\quad&space;R(S_8,&space;RIGHT)&space;=&space;1&space;\quad&space;\&&space;\quad&space;R(S,&space;\cdot)&space;=&space;0&space;\&space;\forall&space;S&space;\neq&space;S_1,&space;S_2" title="R(S_1, LEFT) = 0.1,\quad R(S_8, RIGHT) = 1 \quad \& \quad R(S, \cdot) = 0 \ \forall S \neq S_1, S_2" /></a>
  
  - **P**: <a href="https://www.codecogs.com/eqnedit.php?latex=P(S_{i&plus;1})&space;=&space;1,&space;\&space;s.t.&space;\&space;i&space;\sim&space;Binomial(N&space;=&space;7,&space;p&space;=&space;0.2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(S_{i&plus;1})&space;=&space;1,&space;\&space;s.t.&space;\&space;i&space;\sim&space;Binomial(N&space;=&space;7,&space;p&space;=&space;0.2)" title="P(S_{i+1}) = 1, \ s.t. \ i \sim Binomial(N = 7, p = 0.2)" /></a>
  
  - **γ**: 1
   

- **mdp-v1**: 
  - **S**: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{S_0,&space;S_1,...,&space;S_{10}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\{S_0,&space;S_1,...,&space;S_{9}\}" title="\{S_0, S_1,..., S_{9}\}" /></a>
  
  - **A**: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{LEFT,&space;RIGHT\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\{LEFT,&space;RIGHT\}" title="\{LEFT, RIGHT\}" /></a>
  
  - **T**: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;T(S_i,LEFT)&space;=&space;S_{i-1}&space;\quad&space;\&&space;\quad&space;T(S_i,RIGHT)&space;=&space;S_{i&plus;1}&space;\&space;\forall&space;i&space;\in&space;\{1,&space;2,&space;..,&space;8\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;T(S_i,LEFT)&space;=&space;S_{i-1}&space;\quad&space;\&&space;\quad&space;T(S_i,RIGHT)&space;=&space;S_{i&plus;1}&space;\&space;\forall&space;i&space;\in&space;\{1,&space;2,&space;..,&space;8\}" title="T(S_i,LEFT) = S_{i-1} \quad \& \quad T(S_i,RIGHT) = S_{i+1} \ \forall i \in \{1, 2, .., 8\}" /></a>
  
  - **R**: <a href="https://www.codecogs.com/eqnedit.php?latex=R(S_1,&space;LEFT)&space;=&space;0.1,\quad&space;R(S_8,&space;RIGHT)&space;=&space;1&space;\quad&space;\&&space;\quad&space;R(S,&space;\cdot)&space;=&space;0&space;\&space;\forall&space;S&space;\neq&space;S_1,&space;S_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(S_1,&space;LEFT)&space;=&space;0.1,\quad&space;R(S_8,&space;RIGHT)&space;=&space;1&space;\quad&space;\&&space;\quad&space;R(S,&space;\cdot)&space;=&space;0&space;\&space;\forall&space;S&space;\neq&space;S_1,&space;S_2" title="R(S_1, LEFT) = 0.1,\quad R(S_8, RIGHT) = 1 \quad \& \quad R(S, \cdot) = 0 \ \forall S \neq S_1, S_2" /></a>
  
  - **P**: <a href="https://www.codecogs.com/eqnedit.php?latex=P(S_{i&plus;1})&space;=&space;BinomialPDF(x&space;=&space;i,&space;N&space;=&space;7,&space;p&space;=&space;0.2),&space;\&space;P(S_0)&space;=&space;0,&space;\&space;P(S_9)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(S_{i&plus;1})&space;=&space;BinomialPDF(x&space;=&space;i,&space;N&space;=&space;7,&space;p&space;=&space;0.2),&space;\&space;P(S_0)&space;=&space;0,&space;\&space;P(S_9)&space;=&space;0" title="P(S_{i+1}) = BinomialPDF(x = i, N = 7, p = 0.2), \ P(S_0) = 0, \ P(S_9) = 0" /></a>
  
  - **γ**: 1
   

The MDP chain looks like this. For both the MPDs, the parameter N and p are adjustible.
![MDP transtion](https://i.imgur.com/kSYCUEx.png)
