import tensorflow as tf
import tensorlayer as tl

from rlzoo.common import math_utils
from rlzoo.common.value_networks import *
from rlzoo.common.policy_networks import *
from gym import spaces
from rlzoo.common.utils import set_seed

""" 
full list of algorithm parameters (alg_params)
-----------------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
gamma: discounted factor of reward
action_range: scale of action values
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
save_interval: time steps for saving the weights and plotting the results
mode: 'train' or 'test'
render:  if true, visualize the environment
------------------------------------------------
"""


def atari(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 2e-4  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=500,
        test_episodes=100,
        save_interval=50,
    )

    return alg_params, learn_params


def classic_control(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 2e-4  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=500,
        test_episodes=100,
        save_interval=50,
    )

    return alg_params, learn_params


def box2d(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 2e-4  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=500,
        test_episodes=100,
        save_interval=50,
    )

    return alg_params, learn_params


def mujoco(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 2e-4  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=500,
        test_episodes=100,
        save_interval=50,
    )

    return alg_params, learn_params


def robotics(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 2e-4  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=500,
        test_episodes=100,
        save_interval=50,
    )

    return alg_params, learn_params


def dm_control(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 2e-4  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=500,
        test_episodes=100,
        save_interval=50,
    )

    return alg_params, learn_params


def rlbench(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 2e-4  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=500,
        test_episodes=100,
        save_interval=50,
    )

    return alg_params, learn_params
