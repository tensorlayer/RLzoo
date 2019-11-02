import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *
from gym import spaces
from common.utils import set_seed


def atari(env, default_seed=True):
    if default_seed:
        seed = 2 
        set_seed(seed, env) # reproducible
    
    alg_params = dict(
        gamma=0.9,
        action_range=1  # integer
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200, 
        train_episodes=10000, 
        test_episodes=10, 
        save_interval=100,
    )

    return alg_params, learn_params


def classic_control(env, default_seed=True):
    if default_seed:
        seed = 2 
        set_seed(seed, env) # reproducible
    
    alg_params = dict(
        gamma=0.9,
        action_range=1  # integer because some envs in classic_control are discrete
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200, 
        train_episodes=1000, 
        test_episodes=10, 
        save_interval=100,
    )

    return alg_params, learn_params


def box2d(env, default_seed=True):
    if default_seed:
        seed = 2 
        set_seed(seed, env) # reproducible
    
    alg_params = dict(
        gamma=0.9,
        action_range=1.
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200, 
        train_episodes=1000, 
        test_episodes=10, 
        save_interval=100,
    )

    return alg_params, learn_params


def mujoco(env, default_seed=True):
    if default_seed:
        seed = 2 
        set_seed(seed, env) # reproducible
    
    alg_params = dict(
        gamma=0.9,
        action_range=1.
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200, 
        train_episodes=1000, 
        test_episodes=10, 
        save_interval=100,
    )

    return alg_params, learn_params


def robotics(env, default_seed=True):
    if default_seed:
        seed = 2 
        set_seed(seed, env) # reproducible
    
    alg_params = dict(
        gamma=0.9,
        action_range=1.
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200, 
        train_episodes=1000, 
        test_episodes=10, 
        save_interval=100,
    )

    return alg_params, learn_params


def dm_control(env, default_seed=True):
    if default_seed:
        seed = 2 
        set_seed(seed, env) # reproducible
    
    alg_params = dict(
        gamma=0.9,
        action_range=1.
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200, 
        train_episodes=1000, 
        test_episodes=10, 
        save_interval=100,
    )

    return alg_params, learn_params

def rlbench(env, default_seed=True):
    if default_seed:
        seed = 2 
        set_seed(seed, env) # reproducible
    
    alg_params = dict(
        gamma=0.9,
        action_range=1.
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=200,
        train_episodes=1000, 
        test_episodes=10, 
        save_interval=100,
    )

    return alg_params, learn_params