# Reinforcement Learning Algorithms Zoo
RLzoo is a collection of most practical reinforcement learning algorithms, frameworks and applications. It is implemented with Tensorflow 2.0
and API of neural network layers in TensorLayer 2, to provide a hands-on fast-developing approach for reinforcement learning practices. It supports
basic toy-tests like [OpenAI Gym](https://gym.openai.com/) and [DeepMind Control Suite](https://github.com/deepmind/dm_control) with very simple configurations.
Moreover, RLzoo supports large-scale distributed training framework for more realistic scenarios with [Unity 3D](https://github.com/Unity-Technologies/ml-agents), 
[Mujoco](http://www.mujoco.org/), [Bullet Physics](https://github.com/bulletphysics/bullet3), and robotic learning tasks with [Vrep](http://www.coppeliarobotics.com/)/[Pyrep](https://github.com/stepjam/PyRep), etc.

- [Contents](#contents)
  - [Algorithms](#algorithms)
  - [Applications](#applications)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Citing](#citing)

Please note that this repository using RL algorithms with **high-level API**. So if you want to get familiar with each algorithm more quickly, please look at our **[RL tutorials](https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning)** where each algorithm is implemented individually in a more straightforward manner.

## Status: Work-in-Progress:
Currently the repository is still in development, and there may be some envrionments incompatible with our algorithms. If you find any problems or have any suggestions, feel free to contact with us!

## Contents:
### Algorithms:

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

### Applications:

## Prerequisites:

* python 3.5
* tensorflow >= 2.0.0 or tensorflow-gpu >= 2.0.0a0
* tensorlayer >= 2.0.1
* tensorflow-probability
* tf-nightly-2.0-preview

`pip install -r requirements.txt`

## Usage:

### 1. Implicit Configurations

RL zoo with **implicit configurations** means the configurations for learning are not contained in the main script for running (i.e. `run_rlzoo.py`), but in the `default.py` file in each algorithm folder (for example, `baselines/algorithms/sac/default.py` is the default parameters configuration for SAC algorithm). Whenever you want to change the configurations for learning, including (1) parameter values for the algorithm and learning process, (2) the network structures, (3) the optimizers, etc, you need to go to the `default.py` file under the folder of each algorithm for achieving that. 

#### Common Interface:

```python
EnvName = 'Pendulum-v0'  # chose environment
EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'dm_control'][0]  # select environment type

env = build_env(EnvName, EnvType)  # build environment with wrappers
alg_params, learn_params = call_default_params(env, EnvType, 'TD3')  # call default parameters for the algorithm and learning process
alg = TD3(**alg_params) # instantiate the algorithm
alg.learn(env=env, train_episodes=1000, test_episodes=1000, 
        save_interval=100, mode='train', render=False, **learn_params)  # start the learning process
```

#### To Run:

```
python run_rlzoo.py
```

### 2. Explicit Configurations

RL zoo with **explicit configurations** means the configurations for learning, including parameter values for the algorithm and the learning process, the network structures used in the algorithms and the optimizers etc, are explicitly displayed in the main script for running. And the main scripts are under the folder of each algorithm, for example, `./baselines/algorithms/sac/run_sac.py` can be called with `python algorithms/sac/run_sac.py` from the root file `./baselines/` to run the learning process same as in above implicit configurations.

#### A Quick Example:

```python
''' load environment '''
env = gym.make('CartPole-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
state_shape = env.observation_space.shape
action_shape = (env.action_space.n,)

''' build networks for the algorithm '''
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim = 64 # dimension of hidden layers for the networks
with tf.name_scope('AC'):
        with tf.name_scope('Critic'):
                critic = MlpValueNetwork(state_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
        with tf.name_scope('Actor'):
                actor = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
net_list = [actor, critic]

''' choose optimizers '''
a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
a_optimizer = tf.optimizers.Adam(a_lr)
c_optimizer = tf.optimizers.Adam(c_lr)
optimizers_list=[a_optimizer, c_optimizer]

model=AC(net_list, optimizers_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
gamma: discounted factor of reward
'''

model.learn(env, train_episodes=100, test_episodes=1000, max_steps=1000,
        seed=2, save_interval=100, mode='train', render=False)
''' 
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
seed: random seed
save_interval: timesteps for saving the weights and plotting the results
mode: 'train' or 'test'
render:  if true, visualize the environment
'''


obs = env.reset()
for i in range(100):
    action = model.get_action(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
```

#### To Run:

```python
python algorithms/*ALGORITHM_NAME*/run_*ALGORITHM_NAME*.py 
# for example
python algorithms/ac/run_ac.py
```

## Troubleshooting:

* If you meet the error`AttributeError: module 'tensorflow' has no attribute 'contrib'` when running the code after installing tensorflow-probability, try:
`pip install --upgrade tf-nightly-2.0-preview tfp-nightly`

## Citing:

```
@misc{Reinforcement Learning Algorithms Zoo,
  author = {Zihan Ding, Yanhua Huang, Tianyang Yu, Hongming Zhang, Hao Dong},
  title = {RLzoo},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tensorlayer/RLzoo}},
}
```