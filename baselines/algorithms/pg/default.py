import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.policy_networks import *


def atari(env, set_seed=False):
    if set_seed:
        seed = 2
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        train_episodes=200,
        test_episodes=100,
        max_steps=200,
        save_interval=100,
        gamma=0.95
    )

    return alg_params, learn_params


def classic_control(env, set_seed=False):
    if set_seed:
        seed = 2
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        train_episodes=200,
        test_episodes=100,
        max_steps=200,
        save_interval=100,
        gamma=0.95
    )

    return alg_params, learn_params


def box2d(env, set_seed=False):
    if set_seed:
        seed = 2
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        train_episodes=200,
        test_episodes=100,
        max_steps=200,
        save_interval=100,
        gamma=0.95
    )

    return alg_params, learn_params


def mujoco(env, set_seed=False):
    if set_seed:
        seed = 2
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        train_episodes=200,
        test_episodes=100,
        max_steps=200,
        save_interval=100,
        gamma=0.95
    )

    return alg_params, learn_params


def robotics(env, set_seed=False):
    if set_seed:
        seed = 2
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        train_episodes=200,
        test_episodes=100,
        max_steps=200,
        save_interval=100,
        gamma=0.95
    )

    return alg_params, learn_params


def rlbench(env, set_seed=False):
    if set_seed:
        seed = 2
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('PG'):
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     num_hidden_layer * [hidden_dim])
        net_list = [policy_net]
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        learning_rate = 0.02
        policy_optimizer = tf.optimizers.Adam(learning_rate)
        optimizers_list = [policy_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        train_episodes=200,
        test_episodes=100,
        max_steps=200,
        save_interval=100,
        gamma=0.95
    )

    return alg_params, learn_params
