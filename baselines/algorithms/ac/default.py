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
        gamma = 0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4 #number of hidden layers for the networks
        hidden_dim = 64 # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                    critic = MlpValueNetwork(state_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Actor'):
                    actor = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list=[a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2, 
        max_steps=1000
    )

    return alg_params, learn_params


def classic_control(env):
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
        gamma = 0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4 #number of hidden layers for the networks
        hidden_dim = 64 # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                    critic = MlpValueNetwork(state_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Actor'):
                    actor = DeterministicPolicyNetwork(env.observation_space, env.action_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list=[a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2, 
        max_steps=1000
    )

    return alg_params, learn_params
