from rlzoo.common.policy_networks import *
from rlzoo.common.utils import set_seed

""" 
full list of algorithm parameters (alg_params)
-----------------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: time steps for saving
mode: train or test
render: render each step
gamma: reward decay
-----------------------------------------------
"""


def atari(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        save_interval=20,
        gamma=0.95
    )

    return alg_params, learn_params


def classic_control(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        save_interval=20,
        gamma=0.95
    )

    return alg_params, learn_params


def box2d(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        save_interval=20,
        gamma=0.95
    )

    return alg_params, learn_params


def mujoco(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        save_interval=20,
        gamma=0.95
    )

    return alg_params, learn_params


def robotics(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        save_interval=20,
        gamma=0.95
    )

    return alg_params, learn_params


def dm_control(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        save_interval=20,
        gamma=0.95
    )

    return alg_params, learn_params


def rlbench(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict()

    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
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
        save_interval=20,
        gamma=0.95
    )

    return alg_params, learn_params
