import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *


def classic_control(env):
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n,

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(state_shape, action_shape, num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(env=env, train_episodes=300, test_episodes=200, max_steps=3000, save_interval=100,
            mode='train', render=False, gamma=0.95, seed=2)

    return alg_params, learn_params
