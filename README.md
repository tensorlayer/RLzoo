# Reinforcement Learning Algorithms Zoo
RLzoo is a collection of most practical reinforcement learning algorithms, frameworks and applications. It is implemented with Tensorflow 2.0 and API of neural network layers in TensorLayer 2, to provide a hands-on fast-developing approach for reinforcement learning practices and benchmarks. It supports basic toy-tests like [OpenAI Gym](https://gym.openai.com/) and [DeepMind Control Suite](https://github.com/deepmind/dm_control) with very simple configurations. Moreover, RLzoo supports robot learning benchmark environment [RLBench](https://github.com/stepjam/RLBench) based on  [Vrep](http://www.coppeliarobotics.com/)/[Pyrep](https://github.com/stepjam/PyRep) simulator. Other large-scale distributed training framework for more realistic scenarios with [Unity 3D](https://github.com/Unity-Technologies/ml-agents), 
[Mujoco](http://www.mujoco.org/), [Bullet Physics](https://github.com/bulletphysics/bullet3), etc, will be supported in the future.

<!-- <em>Gym: Atari</em>    <em>Gym: Box2D </em>   <em>Gym: Classic Control </em>  <em>Gym: MuJoCo </em>-->

<img src="https://github.com/tensorlayer/RLzoo/blob/master/gif/atari.gif" height=250 width=210 > <img src="https://github.com/tensorlayer/RLzoo/blob/master/gif/box2d.gif" height=250 width=210 > <img src="https://github.com/tensorlayer/RLzoo/blob/master/gif/classic.gif" height=250 width=210 > <img src="https://github.com/tensorlayer/RLzoo/blob/master/gif/mujoco.gif" height=250 width=210 >

<!-- <em>Gym: Robotics</em>    <em>DeepMind Control Suite </em>   <em>Gym: RLBench </em>  -->

<img src="https://github.com/tensorlayer/RLzoo/blob/master/gif/robotics.gif" height=250 width=210 > <img src="https://github.com/tensorlayer/RLzoo/blob/master/gif/dmcontrol.gif" height=250 width=210 > <img src="https://github.com/tensorlayer/RLzoo/blob/master/gif/rlbench.gif" height=250 width=210 > <img src="https://github.com/tensorlayer/tensorlayer/blob/master/img/tl_transparent_logo.png" height=180 width=210 >






We aim to make it easy to configure for all components within RL, including replacing the networks, optimizers, etc. We also  provide automatically adaptive policies and value functions in the common functions: for the observation space, the vector state or the raw-pixel (image) state are supported automatically according to the shape of the space; for the action space, the discrete action or continuous action are supported automatically according to the shape of the space as well. The deterministic or stochastic property of policy needs to be chosen according to each algorithm. Some environments with raw-pixel based observation (e.g. Atari, RLBench) may be hard to train, be patient and play around with the hyperparameters!

**Table of contents:**

- [Status](#status)
- [Contents](#contents)
  - [Algorithms](#algorithms)
  - [Environments](#environments)
  - [Descriptions](#descriptions)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Citing](#citing)

Please note that this repository using RL algorithms with **high-level API**. So if you want to get familiar with each algorithm more quickly, please look at our **[RL tutorials](https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning)** where each algorithm is implemented individually in a more straightforward manner.

## Status: Release
We are currently open to any suggestions or pull requests from the community to make RLzoo a better repository. Given the scope of this project, we expect there could be some issues over
the coming months after initial release. We will keep improving the potential problems and commit when significant changes are made in the future. Current default hyperparameters for each algorithm and each environment may not be optimal, so you can play around with those hyperparameters to achieve best performances. We will release a version with optimal hyperparameters and benchmark results for all algorithms in the future.

## Contents:
### Algorithms:

| Algorithms      | Papers |
| --------------- | -------|
|**Value-based**||
| Q-learning      | [Technical note: Q-learning. Watkins et al. 1992](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)|
| Deep Q-Network (DQN)| [Human-level control through deep reinforcement learning, Mnih et al. 2015.](https://www.nature.com/articles/nature14236/) |
| Prioritized Experience Replay | [Schaul et al. Prioritized experience replay. Schaul et al. 2015.](https://arxiv.org/abs/1511.05952) |
|Dueling DQN|[Dueling network architectures for deep reinforcement learning. Wang et al. 2015.](https://arxiv.org/abs/1511.06581)|
|Double DQN|[Deep reinforcement learning with double q-learning. Van et al. 2016.](https://arxiv.org/abs/1509.06461)|
|Retrace|[Safe and efficient off-policy reinforcement learning. Munos et al. 2016: ](https://arxiv.org/pdf/1606.02647.pdf)|
|Noisy DQN|[Noisy networks for exploration. Fortunato et al. 2017.](https://arxiv.org/pdf/1706.10295.pdf)|
| Distributed DQN (C51)| [A distributional perspective on reinforcement learning. Bellemare et al. 2017.](https://arxiv.org/pdf/1707.06887.pdf) |
|**Policy-based**||
|REINFORCE(PG) | [Reinforcement learning: An introduction. Sutton et al. 2011.](https://www.cambridge.org/core/journals/robotica/article/robot-learning-edited-by-jonathan-h-connell-and-sridhar-mahadevan-kluwer-boston-19931997-xii240-pp-isbn-0792393651-hardback-21800-guilders-12000-8995/737FD21CA908246DF17779E9C20B6DF6)|
| Trust Region Policy Optimization (TRPO)| [Abbeel et al. Trust region policy optimization. Schulman et al.2015.](https://arxiv.org/pdf/1502.05477.pdf) |
| Proximal Policy Optimization (PPO) | [Proximal policy optimization algorithms. Schulman et al. 2017.](https://arxiv.org/abs/1707.06347) |
|Distributed Proximal Policy Optimization (DPPO)|[Emergence of locomotion behaviours in rich environments. Heess et al. 2017.](https://arxiv.org/abs/1707.02286)|
|**Actor-Critic**||
|Actor-Critic (AC)| [Actor-critic algorithms. Konda er al. 2000.](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)|
| Asynchronous Advantage Actor-Critic (A3C)| [Asynchronous methods for deep reinforcement learning. Mnih et al. 2016.](https://arxiv.org/pdf/1602.01783.pdf) |
| Deep Deterministic Policy Gradient (DDPG) | [Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf) |
|Twin Delayed DDPG (TD3)|[Addressing function approximation error in actor-critic methods. Fujimoto et al. 2018.](https://arxiv.org/pdf/1802.09477.pdf)|
|Soft Actor-Critic (SAC)|[Soft actor-critic algorithms and applications. Haarnoja et al. 2018.](https://arxiv.org/abs/1812.05905)|

### Environments:

* [**OpenAI Gym**](https://gym.openai.com/):  

    * Atari
    * Box2D
    * Classic control
    * MuJoCo
    * Robotics

    Full list of environments is [here](https://gym.openai.com/envs/#classic_control).

    List of environments with types of spaces for Atari, Box2D and Classic Control is [here](https://github.com/openai/gym/wiki/Table-of-environments).

* [**DeepMind Control Suite**](https://github.com/deepmind/dm_control):

    The [dm2gym](https://github.com/zuoxingdong/dm2gym) is needed for registering environments in DeepMind Control Suite as Gym environments.

* [**RLBench**](https://github.com/stepjam/RLBench): 

    Full list of environments is [here](https://github.com/stepjam/RLBench/blob/master/rlbench/tasks/__init__.py). 

    Installation of Vrep->PyRep->RLBench follows [here](http://www.coppeliarobotics.com/downloads.html)->[here](https://github.com/stepjam/PyRep)->[here](https://github.com/stepjam/RLBench).

**Note**:

* Make sure the name of environment matches the type of environment in the main script. The types of environments include: 'atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'.
* When using the DeepMind Control Suite, install the [dm2gym](https://github.com/zuoxingdong/dm2gym) package with: `pip install dm2gym`

* When using the RLBench environments, please add the path of your local rlbench repository to python: 
  ```export PYTHONPATH=PATH_TO_YOUR_LOCAL_RLBENCH_REPO```
* A dictionary of all different environments is stored in `./rlzoo/common/env_list.py`

## Descriptions:
The supported configurations for RL algorithms with corresponding environments in RLzoo are listed in the following table.

| Algorithms                 | Action Space        | Policy        | Update     | Envs                                                         |
| -------------------------- | ------------------- | ------------- | ---------- | ------------------------------------------------------------ |
| DQN (double, dueling, PER) | Discrete Only       | --            | Off-policy | Atari, Classic Control                                       |
| AC                         | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |
| PG                         | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |
| DDPG                       | Continuous          | Deterministic | Off-policy | Classic Control, Box2D, Mujoco, Robotics, DeepMind Control, RLBench |
| TD3                        | Continuous          | Deterministic | Off-policy | Classic Control, Box2D, Mujoco, Robotics, DeepMind Control, RLBench |
| SAC                        | Continuous          | Stochastic    | Off-policy | Classic Control, Box2D, Mujoco, Robotics, DeepMind Control, RLBench |
| A3C                        | Discrete/Continuous | Stochastic    | On-policy  | Atari, Classic Control, Box2D, Mujoco, Robotics, DeepMind Control |
| PPO                        | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |
| DPPO                       | Discrete/Continuous | Stochastic    | On-policy  | Atari, Classic Control, Box2D, Mujoco, Robotics, DeepMind Control |
| TRPO                       | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |


## Prerequisites:

* python >=3.5 (python 3.6 is needed if using dm_control)
* tensorflow >= 2.0.0 or tensorflow-gpu >= 2.0.0a0
* tensorlayer >= 2.0.1
* tensorflow-probability
* tf-nightly-2.0-preview
* [Mujoco 2.0](http://www.mujoco.org/), [dm_control](https://github.com/deepmind/dm_control), [dm2gym](https://github.com/zuoxingdong/dm2gym) (if using DeepMind Control Suite environments)
* Vrep, PyRep, RLBench (if using RLBench environments, follows [here](http://www.coppeliarobotics.com/downloads.html), [here](https://github.com/stepjam/PyRep) and [here](https://github.com/stepjam/RLBench))

## Installation:

To install RLzoo package with key requirements:

```
pip install rlzoo
```

## Usage:

### 0. Quick Start
Choose whatever environments with whatever RL algorithms supported in RLzoo, and enjoy the game by running following example in the root file of installed package:
```python
# in the root folder of rlzoo package
cd rlzoo
python run_rlzoo.py
```
The main script `run_rlzoo.py` follows (almost) the same structure for all algorithms on all environments, see the [**full list of examples**](./examples.md).

**General Descriptions:**
RLzoo provides at least two types of interfaces for running the learning algorithms, with (1) implicit configurations or (2) explicit configurations. Both of them start learning program through running a python script, instead of running a long command line with all configurations shortened to be arguments of it (e.g. in Openai Baseline). Our approaches are found to be more interpretable, flexible and convenient to apply in practice. According to the level of explicitness of learning configurations, we provided two different ways of setting learning configurations in python scripts: the first one with implicit configurations uses a `default.py` script to record all configurations for each algorithm, while the second one with explicit configurations exposes all configurations to the running scripts. Both of them can run any RL algorithms on any environments supported in our repository with a simple command line.

### 1. Implicit Configurations

RLzoo with **implicit configurations** means the configurations for learning are not explicitly contained in the main script for running (i.e. `run_rlzoo.py`), but in the `default.py` file in each algorithm folder (for example, `rlzoo/algorithms/sac/default.py` is the default parameters configuration for SAC algorithm). All configurations include (1) parameter values for the algorithm and learning process, (2) the network structures, (3) the optimizers, etc, are divided into configurations for the algorithm (stored in `alg_params`) and configurations for the learning process (stored in `learn_params`). Whenever you want to change the configurations for the algorithm or learning process, you can either go to the folder of each algorithm and modify parameters in `default.py`, or change the values in `alg_params` (a dictionary of configurations for the algorithm) and `learn_params` (a dictionary of configurations for the learning process) in `run_rlzoo.py` according to the keys. 

#### Common Interface:

```python
from rlzoo.common.env_wrappers import build_env
from rlzoo.common.utils import call_default_params
from rlzoo.algorithms import *
# choose algorithm
AlgName = 'TD3'
# chose environment
EnvName = 'Pendulum-v0'  
# select corresponding environment type
EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'robotics', 'dm_control', 'rlbench'][0] 
# build environment with wrappers
env = build_env(EnvName, EnvType)  
# call default parameters for the algorithm and learning process
alg_params, learn_params = call_default_params(env, EnvType, AlgName)  
# instantiate the algorithm
alg = eval(AlgName+'(**alg_params)')
# start the training process
alg.learn(env=env, mode='train', render=False, **learn_params)  
# test after training 
alg.learn(env=env, mode='test', render=True, **learn_params)  
```

#### To Run:

```python
# in the root folder of rlzoo package
cd rlzoo
python run_rlzoo.py
```

### 2. Explicit Configurations

RLzoo with **explicit configurations** means the configurations for learning, including parameter values for the algorithm and the learning process, the network structures used in the algorithms and the optimizers etc, are explicitly displayed in the main script for running. And the main scripts for demonstration are under the folder of each algorithm, for example, `./rlzoo/algorithms/sac/run_sac.py` can be called with `python algorithms/sac/run_sac.py` from the file `./rlzoo` to run the learning process same as in above implicit configurations.

#### A Quick Example:

```python
import gym
from rlzoo.common.utils import make_env, set_seed
from rlzoo.algorithms import AC
from rlzoo.common.value_networks import ValueNetwork
from rlzoo.common.policy_networks import StochasticPolicyNetwork

''' load environment '''
env = gym.make('CartPole-v0').unwrapped
obs_space = env.observation_space
act_space = env.action_space
# reproducible
seed = 2
set_seed(seed, env)

''' build networks for the algorithm '''
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim = 64 # dimension of hidden layers for the networks
with tf.name_scope('AC'):
        with tf.name_scope('Critic'):
            	# choose the critic network, can be replaced with customized network
                critic = ValueNetwork(obs_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
        with tf.name_scope('Actor'):
            	# choose the actor network, can be replaced with customized network
                actor = StochasticPolicyNetwork(obs_space, act_space, hidden_dim_list=num_hidden_layer * [hidden_dim], output_activation=tf.nn.tanh)
net_list = [actor, critic] # list of the networks

''' choose optimizers '''
a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
a_optimizer = tf.optimizers.Adam(a_lr)
c_optimizer = tf.optimizers.Adam(c_lr)
optimizers_list=[a_optimizer, c_optimizer]  # list of optimizers

# intialize the algorithm model, with algorithm parameters passed in
model = AC(net_list, optimizers_list)
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
gamma: discounted factor of reward
action_range: scale of action values
'''

# start the training process, with learning parameters passed in
model.learn(env, train_episodes=500,  max_steps=200,
            save_interval=50, mode='train', render=False)
''' 
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
save_interval: time steps for saving the weights and plotting the results
mode: 'train' or 'test'
render:  if true, visualize the environment
'''

# test after training
model.learn(env, test_episodes=100, max_steps=200,  mode='test', render=True)
```

#### To Run:

In the package folder, we provides examples with explicit configurations for each algorithm. 

```python
# in the root folder of rlzoo package
cd rlzoo
python algorithms/<ALGORITHM_NAME>/run_<ALGORITHM_NAME>.py 
# for example: run actor-critic
python algorithms/ac/run_ac.py
```

## Troubleshooting:

* If you meet the error *'AttributeError: module 'tensorflow' has no attribute 'contrib''* when running the code after installing tensorflow-probability, try:
  `pip install --upgrade tf-nightly-2.0-preview tfp-nightly`
* When trying to use RLBench environments, *'No module named rlbench'* can be caused by no RLBench package installed at your local or a mistake in the python path. You should add `export PYTHONPATH=/home/quantumiracle/research/vrep/PyRep/RLBench` every time you try to run the learning script with RLBench environment or add it to you `~/.bashrc` file once for all.
* If you meet the error that the Qt platform is not loaded correctly when using DeepMind Control Suite environments, it's probably caused by your Ubuntu system not being version 14.04 or 16.04. Check [here](https://github.com/deepmind/dm_control).

## Citing:

```
@misc{Reinforcement Learning Algorithms Zoo,
  author = {Zihan Ding, Tianyang Yu, Yanhua Huang, Hongming Zhang, Hao Dong},
  title = {RLzoo},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tensorlayer/RLzoo}},
}
```

