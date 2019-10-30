import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *


def classic_control(env):

    alg_params = dict(
        replay_buffer_size=10000,
        tau=0.01, var=3
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 30  # dimension of hidden layers for the networks
        with tf.name_scope('DDPG'):
            with tf.name_scope('Q_Net'):
                q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net'):
                target_q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Policy'):
                target_policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space,
                                                               num_hidden_layer * [hidden_dim])

        net_list = [q_net, target_q_net, policy_net, target_policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-3
        critic_lr = 2e-3
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(gamma=0.9,
                        seed=1,
                        batch_size=32)

    return alg_params, learn_params
