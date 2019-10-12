import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *
from gym import spaces

def atari(env):
    state_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        action_range = 1
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=3000, 
        gamma=0.95, 
        seed=2)

    return alg_params, learn_params


def classic_control(env):
    state_shape = env.observation_space.shape
    if isinstance(env.action_space, spaces.Discrete):
        action_shape = (env.action_space.n,)
        action_range = 1  # int
    elif isinstance(env.action_space, spaces.Box):
        assert len(env.action_space.shape) == 1
        action_shape = env.action_space.shape
        action_range = 1.  # float
    else:
        raise NotImplementedError

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        action_range = action_range
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=3000, 
        gamma=0.95, 
        seed=2)

    return alg_params, learn_params

def box2d(env):
    state_shape = env.observation_space.shape
    if isinstance(env.action_space, spaces.Discrete):
        action_shape = (env.action_space.n,)
        action_range = 1  # int
    elif isinstance(env.action_space, spaces.Box):
        assert len(env.action_space.shape) == 1
        action_shape = env.action_space.shape
        action_range = 1.  # float
    else:
        raise NotImplementedError

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        action_range = action_range
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=3000, 
        gamma=0.95, 
        seed=2)

    return alg_params, learn_params


def mujoco(env):
    state_shape = env.observation_space.shape
    if isinstance(env.action_space, spaces.Discrete):
        action_shape = (env.action_space.n,)
        action_range = 1  # int
    elif isinstance(env.action_space, spaces.Box):
        assert len(env.action_space.shape) == 1
        action_shape = env.action_space.shape
        action_range = 1.  # float
    else:
        raise NotImplementedError

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        action_range = action_range
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=3000, 
        gamma=0.95, 
        seed=2)

    return alg_params, learn_params


def robotics(env):
    state_shape = env.observation_space.shape
    if isinstance(env.action_space, spaces.Discrete):
        action_shape = (env.action_space.n,)
        action_range = 1  # int
    elif isinstance(env.action_space, spaces.Box):
        assert len(env.action_space.shape) == 1
        action_shape = env.action_space.shape
        action_range = 1.  # float
    else:
        raise NotImplementedError

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        action_range = action_range
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=3000, 
        gamma=0.95, 
        seed=2)

    return alg_params, learn_params


def rlbench(env):
    state_shape = env.observation_space.shape
    if isinstance(env.action_space, spaces.Discrete):
        action_shape = (env.action_space.n,)
    elif isinstance(env.action_space, spaces.Box):
        assert len(env.action_space.shape) == 1
        action_shape = env.action_space.shape
    else:
        raise NotImplementedError

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        action_range = 0.1
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=100, 
        gamma=0.95, 
        seed=2)

    return alg_params, learn_params
