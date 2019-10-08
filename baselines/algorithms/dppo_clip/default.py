import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *


def classic_control(env):
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_max = env.action_space.high
    action_min = env.action_space.low

    alg_params = dict(
        state_dim=state_shape[0],
        action_dim=action_shape[0],
        a_bounds=[action_min, action_max],
        epsilon=0.2
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 100  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO_CLIP'):
            with tf.name_scope('V_Net'):
                v_net = MlpValueNetwork(state_shape, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(state_shape, action_shape, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Old_Policy'):
                old_policy_net = StochasticPolicyNetwork(state_shape, action_shape, [hidden_dim] * num_hidden_layer,
                                                         trainable=False)

        net_list = [v_net, policy_net, old_policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(env=env, train_episodes=1000, test_episodes=10, max_steps=200, save_interval=10,
                        mode='train', gamma=0.9, a_update_steps=10, c_update_steps=10, n_worker=4,
                        batch_size=32, seed=1, reward_shaping=lambda x: (x + 8) / 8)

    return alg_params, learn_params
