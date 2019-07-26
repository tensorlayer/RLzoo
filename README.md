# Reinforcement Learning Algorithms Zoo
RLzoo is a collection of most practical reinforcement learning algorithms, frameworks and applications. It is implemented with Tensorflow 2.0
and API of neural network layers in TensorLayer 2, to provide a hands-on fast-developing approach for reinforcement learning practices. It supports
basic toy-tests like [OpenAI Gym](https://gym.openai.com/) and [DeepMind Control Suite](https://github.com/deepmind/dm_control) with very simple configurations.
Moreover, RLzoo supports large-scale distributed training framework for more realistic scenarios with [Unity 3D](https://github.com/Unity-Technologies/ml-agents), 
[Mujoco](http://www.mujoco.org/), [Bullet Physics](https://github.com/bulletphysics/bullet3), and robotic learning tasks with [Vrep](http://www.coppeliarobotics.com/)/[Pyrep](https://github.com/stepjam/PyRep), etc.

- [Contents](#contents)
  - [Algorithms](##1. algorithms)
  - [Applications](##2. applications)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Citing](#citing)

## Contents:
### 1. Algorithms:

### value-based
| Algorithms      | Action Space | Tutorial Env   | Papers |
| --------------- | ------------ | -------------- | -------|
|**value-based**||||
| Q-learning      | Discrete     | FrozenLake     | [Technical note: Q-learning. Watkins et al. 1992](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)|
| Deep Q-Network (DQN)| Discrete     | FrozenLake     | [Human-level control through deep reinforcement learning, Mnih et al. 2015.](https://www.nature.com/articles/nature14236/) |
| Prioritized Experience Replay | Discrete     | Pong, CartPole | [Schaul et al. Prioritized experience replay. Schaul et al. 2015.](https://arxiv.org/abs/1511.05952) |
|Dueling DQN|Discrete     | Pong, CartPole |[Dueling network architectures for deep reinforcement learning. Wang et al. 2015.](https://arxiv.org/abs/1511.06581)|
|Double DQN| Discrete     | Pong, CartPole |[Deep reinforcement learning with double q-learning. Van et al. 2016.](https://arxiv.org/abs/1509.06461)|
|Retrace|Discrete     | Pong, CartPole |[Safe and efficient off-policy reinforcement learning. Munos et al. 2016: ](https://arxiv.org/pdf/1606.02647.pdf)|
|Noisy DQN|Discrete     | Pong, CartPole |[Noisy networks for exploration. Fortunato et al. 2017.](https://arxiv.org/pdf/1706.10295.pdf)|
| Distributed DQN (C51)| Discrete     | Pong, CartPole | [A distributional perspective on reinforcement learning. Bellemare et al. 2017.](https://arxiv.org/pdf/1707.06887.pdf) |
|**policy-based**||||
|REINFORCE(PG) |Discrete/Continuous|CartPole | [Reinforcement learning: An introduction. Sutton et al. 2011.](https://www.cambridge.org/core/journals/robotica/article/robot-learning-edited-by-jonathan-h-connell-and-sridhar-mahadevan-kluwer-boston-19931997-xii240-pp-isbn-0792393651-hardback-21800-guilders-12000-8995/737FD21CA908246DF17779E9C20B6DF6)|
| Trust Region Policy Optimization (TRPO)| Discrete/Continuous | Pendulum | [Abbeel et al. Trust region policy optimization. Schulman et al.2015.](https://arxiv.org/pdf/1502.05477.pdf) |
| Proximal Policy Optimization (PPO) |Discrete/Continuous |Pendulum| [Proximal policy optimization algorithms. Schulman et al. 2017.](https://arxiv.org/abs/1707.06347) |
|Distributed Proximal Policy Optimization (DPPO)|Discrete/Continuous |Pendulum|[Emergence of locomotion behaviours in rich environments. Heess et al. 2017.](https://arxiv.org/abs/1707.02286)|
|**actor-critic**||||
|Actor-Critic (AC)|Discrete/Continuous|CartPole| [Actor-critic algorithms. Konda er al. 2000.](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)|
| Asynchronous Advantage Actor-Critic (A3C)| Discrete/Continuous | BipedalWalker| [Asynchronous methods for deep reinforcement learning. Mnih et al. 2016.](https://arxiv.org/pdf/1602.01783.pdf) |
| DDPG|Discrete/Continuous |Pendulum| [Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf) |
|TD3|Discrete/Continuous |Pendulum|[Addressing function approximation error in actor-critic methods. Fujimoto et al. 2018.](https://arxiv.org/pdf/1802.09477.pdf)|
|Soft Actor-Critic (SAC)|Discrete/Continuous |Pendulum|[Soft actor-critic algorithms and applications. Haarnoja et al. 2018.](https://arxiv.org/abs/1812.05905)|

### 2. Applications:

## Prerequisites:

* python 3.5
* tensorflow >= 2.0.0 or tensorflow-gpu >= 2.0.0a0
* tensorlayer >= 2.0.1
* tensorflow-probability
* tf-nightly-2.0-preview

`pip install -r requirements.txt`

## Usage:
`python main.py --env=Pendulum-v0 --algorithm=td3 --train_episodes=600 --mode=train`

`python main.py --env=BipedalWalker-v2 --algorithm=a3c --train_episodes=600 --mode=train --number_workers=2`

`python main.py --env=CartPole-v0 --algorithm=ac --train_episodes=600 --mode=train`

`python main.py --env=FrozenLake-v0 --algorithm=dqn --train_episodes=6000 --mode=train`

## Troubleshooting:

* If you meet the error`AttributeError: module 'tensorflow' has no attribute 'contrib'` when running the code after installing tensorflow-probability, try:
`pip install --upgrade tf-nightly-2.0-preview tfp-nightly`

## Citing:
