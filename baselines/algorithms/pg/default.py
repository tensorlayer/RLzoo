import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *


def atari(env):
    state_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        reward_decay = 0.95,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2 #number of hidden layers for the networks
        hidden_dim=32 # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            policy = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
        net_list = [policy]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        lr = 0.01  # lr: learning rate of the policy
        optimizer = tf.optimizers.Adam(lr)
        optimizers_list = [optimizer]

    learn_params = dict(
        gamma=0.99,
        seed=2, 
        max_steps=1000
    )

    return alg_params, learn_params


def classic_control(env):
    state_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        reward_decay = 0.95,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2 #number of hidden layers for the networks
        hidden_dim=32 # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            policy = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
        net_list = [policy]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        lr = 0.01  # lr: learning rate of the policy
        optimizer = tf.optimizers.Adam(lr)
        optimizers_list = [optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        gamma=0.99,
        seed=2, 
        max_steps=1000
    )
    return alg_params, learn_params
