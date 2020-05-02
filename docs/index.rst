.. RLzoo documentation master file, created by
   sphinx-quickstart on Wed Apr 29 23:00:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reinforcement Learning Zoo for Simple Usage
============================================

.. image:: img/logo.png
  :width: 50 %
  :align: center
  :target: https://github.com/tensorlayer/rlzoo

RLzoo is a collection of the most practical reinforcement learning algorithms, frameworks and applications. It is implemented with Tensorflow 2.0 and API of neural network layers in TensorLayer 2, to provide a hands-on fast-developing approach for reinforcement learning practices and benchmarks. It supports basic toy-test environments like `OpenAI Gym <https://gym.openai.com/>`_ and `DeepMind Control Suite <https://github.com/deepmind/dm_control>`_ with very simple configurations. Moreover, RLzoo supports robot learning benchmark environment `RLBench <https://github.com/stepjam/RLBench>`_ based on Vrep/Pyrep simulator. Other large-scale distributed training framework for more realistic scenarios with Unity 3D, Mujoco, Bullet Physics, etc, will be supported in the future. A `Springer textbook <https://deepreinforcementlearningbook.org>`__ is also provided, you can get the free PDF if your institute has Springer license.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   guide/quickstart
   guide/configuration

.. toctree::
   :maxdepth: 1
   :caption: RL Algorithms

   algorithms/dqn
   algorithms/pg
   algorithms/ac
   algorithms/a3c
   algorithms/ddpg
   algorithms/td3
   algorithms/sac
   algorithms/trpo
   algorithms/ppo
   algorithms/dppo

.. toctree::
   :maxdepth: 1
   :caption: Common

   common/basicnets
   common/policynets
   common/valuenets
   common/buffer
   common/distributions
   common/envwrappers
   common/envlist
   common/mathutils
   common/utils

.. toctree::
   :maxdepth: 1
   :caption: Other Resources

   other/drl_book
   other/drl_tutorial

Contributing
==================

This project is under active development, if you want to join the core team, feel free to contact Zihan Ding at ....

Citation
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. image:: img/logo.png
  :width: 70 %
  :align: center
  :target: https://github.com/tensorlayer/rlzoo

