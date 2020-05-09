from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
from rlzoo.common.utils import set_seed

""" 
full list of algorithm parameters (alg_params)
-----------------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
damping_coeff: Artifact for numerical stability
cg_iters: Number of iterations of conjugate gradient to perform
delta: KL-divergence limit for TRPO update.
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: time steps for saving
gamma: reward discount factor
mode: train or test
render: render each step
batch_size: update batch size
backtrack_iters: Maximum number of steps allowed in the backtracking line search
backtrack_coeff: How far back to step during backtracking line search
train_critic_iters: critic update iteration steps
-----------------------------------------------
"""


def atari(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
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

    learn_params = dict(train_episodes=2000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=100,
                        gamma=0.9,
                        batch_size=256,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params


def classic_control(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
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

    learn_params = dict(train_episodes=2000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=100,
                        gamma=0.9,
                        batch_size=256,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params


def box2d(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
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

    learn_params = dict(train_episodes=2000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=100,
                        gamma=0.9,
                        batch_size=256,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params


def mujoco(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
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

    learn_params = dict(train_episodes=2000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=100,
                        gamma=0.9,
                        batch_size=256,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params


def robotics(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
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

    learn_params = dict(train_episodes=2000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=100,
                        gamma=0.9,
                        batch_size=256,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params


def dm_control(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
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

    learn_params = dict(train_episodes=2000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=100,
                        gamma=0.9,
                        batch_size=256,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params


def rlbench(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
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

    learn_params = dict(train_episodes=2000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=100,
                        gamma=0.9,
                        batch_size=256,
                        backtrack_iters=10,
                        backtrack_coeff=0.8,
                        train_critic_iters=80)

    return alg_params, learn_params
