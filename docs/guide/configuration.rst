Configurations Overview
=================================

Supported DRL Agorithms
--------------------------
Generally RLzoo supports following DRL algorithms:

**Value-based methods**

* `Deep Q-Networks (DQN) <https://www.nature.com/articles/nature14236/>`_
* `Double DQN <https://arxiv.org/abs/1509.06461>`_
* `Dueling DQN <https://arxiv.org/abs/1511.06581>`_
* `Prioritized Experience Replay (PER) <https://arxiv.org/abs/1511.05952>`_
* `Retrace <https://arxiv.org/pdf/1606.02647.pdf>`_
* `Noisy DQN <https://arxiv.org/pdf/1706.10295.pdf>`_
* `Distributed DQN <https://arxiv.org/pdf/1707.06887.pdf>`_

**Policy-based methods**

* `Vanilla Policy Gradient (VPG) <https://link.springer.com/article/10.1007/BF00992696>`_
* `Trust Region Policy Optimization (TRPO) <https://arxiv.org/pdf/1502.05477.pdf>`_
* `Proximal Policy Optimization (PPO) <https://arxiv.org/abs/1707.06347>`_
* `Distributed PPO (DPPO) <https://arxiv.org/abs/1707.02286>`_

**Actor-critic methods**

* `Actor-Critic (AC) <https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf>`_
* `Asychronous Advantage Actor-Critic (A3C) <https://arxiv.org/pdf/1602.01783.pdf>`_
* `Deep Deterministic Policy Gradient (DDPG) <https://arxiv.org/pdf/1509.02971.pdf>`_
* `Twin Delayed DDPG (TD3) <https://arxiv.org/pdf/1802.09477.pdf>`_
* `Soft Actor-Critic (SAC) <https://arxiv.org/abs/1812.05905>`_


Supported Environments
--------------------------
Generally RLzoo supports following environments for DRL:

* `OpenAI Gym <https://gym.openai.com/>`_
    * Atari
    * Box2D
    * Classic Control 
    * MuJoCo
    * Robotics
* `DeepMind Control Suite <https://github.com/deepmind/dm_control>`_

* `RLBench <https://github.com/stepjam/RLBench>`_


Full list of specific names of environments supported in RLzoo can be checked in :ref:`env_list`.

Supported Configurations
-----------------------------
Not all configurations (specific RL algorithm on specific environment) are supported in RLzoo, as in other libraries. The supported configurations for RL algorithms with corresponding environments in RLzoo are listed in the following table.

+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| Algorithms                 | Action Space        | Policy        | Update     | Envs                                                                |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| DQN (double, dueling, PER) | Discrete Only       | NA            | Off-policy | Atari, Classic Control                                              |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| AC                         | Discrete/Continuous | Stochastic    | On-policy  | All                                                                 |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| PG                         | Discrete/Continuous | Stochastic    | On-policy  | All                                                                 |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| DDPG                       | Continuous          | Deterministic | Off-policy | Classic Control, Box2D, MuJoCo, Robotics, DeepMind Control, RLBench |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| TD3                        | Continuous          | Deterministic | Off-policy | Classic Control, Box2D, MuJoCo, Robotics, DeepMind Control, RLBench |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| SAC                        | Continuous          | Stochastic    | Off-policy | Classic Control, Box2D, MuJoCo, Robotics, DeepMind Control, RLBench |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| A3C                        | Discrete/Continuous | Stochastic    | On-policy  | Atari, Classic Control, Box2D, MuJoCo, Robotics, DeepMind Control   |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| PPO                        | Discrete/Continuous | Stochastic    | On-policy  | All                                                                 |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| DPPO                       | Discrete/Continuous | Stochastic    | On-policy  | Atari, Classic Control, Box2D, MuJoCo, Robotics, DeepMind Control   |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+
| TRPO                       | Discrete/Continuous | Stochastic    | On-policy  | All                                                                 |
+----------------------------+---------------------+---------------+------------+---------------------------------------------------------------------+