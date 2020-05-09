from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
from rlzoo.common.utils import set_seed

""" 
full list of algorithm parameters (alg_params)
-----------------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
replay_buffer_size: the size of buffer for storing explored samples
tau: soft update factor
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: time steps for saving
explore_steps: for random action sampling in the beginning of training
mode: train or test mode
render: render each step
batch_size: update batch size
gamma: reward decay factor
noise_scale: range of action noise for exploration
noise_scale_decay: noise scale decay factor
-----------------------------------------------
"""

def classic_control(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
        set_seed(seed, env)

    alg_params = dict(
        replay_buffer_size=10000,
        tau=0.01,
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DDPG'):
            with tf.name_scope('Q_Net'):
                q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net'):
                target_q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space,
                                                        num_hidden_layer * [hidden_dim])
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

    learn_params = dict(
        train_episodes=100,
        test_episodes=10,
        max_steps=200,
        save_interval=10,
        explore_steps=500,
        batch_size=32,
        gamma=0.9,
        noise_scale=1.,
        noise_scale_decay=0.995
    )

    return alg_params, learn_params


def box2d(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
        set_seed(seed, env)

    alg_params = dict(
        replay_buffer_size=10000,
        tau=0.01,
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DDPG'):
            with tf.name_scope('Q_Net'):
                q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net'):
                target_q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space,
                                                        num_hidden_layer * [hidden_dim])
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

    learn_params = dict(
        train_episodes=100,
        test_episodes=10,
        max_steps=200,
        save_interval=10,
        explore_steps=500,
        batch_size=32,
        gamma=0.9,
        noise_scale=1.,
        noise_scale_decay=0.995
    )

    return alg_params, learn_params


def mujoco(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
        set_seed(seed, env)

    alg_params = dict(
        replay_buffer_size=10000,
        tau=0.01,
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DDPG'):
            with tf.name_scope('Q_Net'):
                q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net'):
                target_q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space,
                                                        num_hidden_layer * [hidden_dim])
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

    learn_params = dict(
        train_episodes=100,
        test_episodes=10,
        max_steps=200,
        save_interval=10,
        explore_steps=500,
        batch_size=32,
        gamma=0.9,
        noise_scale=1.,
        noise_scale_decay=0.995
    )

    return alg_params, learn_params


def robotics(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
        set_seed(seed, env)

    alg_params = dict(
        replay_buffer_size=10000,
        tau=0.01,
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DDPG'):
            with tf.name_scope('Q_Net'):
                q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net'):
                target_q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space,
                                                        num_hidden_layer * [hidden_dim])
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

    learn_params = dict(
        train_episodes=100,
        test_episodes=10,
        max_steps=200,
        save_interval=10,
        explore_steps=500,
        batch_size=32,
        gamma=0.9,
        noise_scale=1.,
        noise_scale_decay=0.995
    )

    return alg_params, learn_params


def dm_control(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
        set_seed(seed, env)

    alg_params = dict(
        replay_buffer_size=10000,
        tau=0.01,
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DDPG'):
            with tf.name_scope('Q_Net'):
                q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net'):
                target_q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space,
                                                        num_hidden_layer * [hidden_dim])
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

    learn_params = dict(
        train_episodes=100,
        test_episodes=10,
        max_steps=200,
        save_interval=10,
        explore_steps=500,
        batch_size=32,
        gamma=0.9,
        noise_scale=1.,
        noise_scale_decay=0.995
    )

    return alg_params, learn_params


def rlbench(env, default_seed=True):
    if default_seed:
        # reproducible
        seed = 2
        set_seed(seed, env)

    alg_params = dict(
        replay_buffer_size=1000,
        tau=0.01,
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DDPG'):
            with tf.name_scope('Q_Net'):
                q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net'):
                target_q_net = QNetwork(env.observation_space, env.action_space, num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = DeterministicPolicyNetwork(env.observation_space, env.action_space,
                                                        num_hidden_layer * [hidden_dim])
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

    learn_params = dict(
        train_episodes=100,
        test_episodes=10,
        max_steps=200,
        save_interval=10,
        explore_steps=500,
        batch_size=32,
        gamma=0.9,
        noise_scale=1.,
        noise_scale_decay=0.995
    )

    return alg_params, learn_params