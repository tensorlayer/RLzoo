import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *
from common.utils import set_seed


def classic_control(env, default_seed=True):

    if default_seed:
        # reproducible
        seed = 1
        set_seed(seed, env)

    alg_params = dict(
        damping_coeff=0.1,
        cg_iters=10,
        delta=0.01
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('TRPO'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     [hidden_dim] * num_hidden_layer, output_activation=tf.nn.tanh)

        net_list = [v_net, policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        critic_lr = 1e-3
        optimizers_list = [tf.optimizers.Adam(critic_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=200,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=10,
                        gamma=0.9,
                        batch_size=32,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params
