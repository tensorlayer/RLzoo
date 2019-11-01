import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *
from common.utils import set_seed


def classic_control(env, default_seed=True):
    if default_seed:
        seed = 1
        set_seed(seed)

    alg_params = dict(
        epsilon=0.2
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 100  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO_CLIP'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space, [hidden_dim] * num_hidden_layer)

        net_list = v_net, policy_net
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=1000,
                        test_episodes=10,
                        max_steps=200,
                        save_interval=10,
                        gamma=0.9,
                        a_update_steps=10,
                        c_update_steps=10,
                        n_worker=4,
                        batch_size=32)

    return alg_params, learn_params
