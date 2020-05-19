from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
from rlzoo.common.utils import set_seed

""" 
full list of algorithm parameters (alg_params)
-----------------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
replay_buffer_capacity: the size of buffer for storing explored samples
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
batch_size:  udpate batchsize
explore_steps:  for random action sampling in the beginning of training
update_itr: repeated updates for single step
policy_target_update_interval: delayed update for the policy network and target networks
reward_scale: value range of reward
save_interval: timesteps for saving the weights and plotting the results
mode: 'train'  or 'test'
AUTO_ENTROPY: automatically udpating variable alpha for entropy
render: if true, visualize the environment
-----------------------------------------------
"""


def classic_control(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        replay_buffer_capacity=5e5,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                     output_activation=None,
                                                     state_conditioned=True)
        net_list = [soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2, policy_net]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        soft_q_lr, policy_lr, alpha_lr = 3e-4, 3e-4, 3e-4  # soft_q_lr: learning rate of the Q network; policy_lr: learning rate of the policy network; alpha_lr: learning rate of the variable alpha
        soft_q_optimizer1 = tf.optimizers.Adam(soft_q_lr)
        soft_q_optimizer2 = tf.optimizers.Adam(soft_q_lr)
        policy_optimizer = tf.optimizers.Adam(policy_lr)
        alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        optimizers_list = [soft_q_optimizer1, soft_q_optimizer2, policy_optimizer, alpha_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=150,
        batch_size=64,
        explore_steps=200,
        update_itr=3,
        policy_target_update_interval=3,
        reward_scale=1.,
        AUTO_ENTROPY=True,
        train_episodes=100,
        test_episodes=10,
        save_interval=10,
    )

    return alg_params, learn_params


def box2d(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        replay_buffer_capacity=5e5,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                     output_activation=None,
                                                     state_conditioned=True)
        net_list = [soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2, policy_net]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        soft_q_lr, policy_lr, alpha_lr = 3e-4, 3e-4, 3e-4  # soft_q_lr: learning rate of the Q network; policy_lr: learning rate of the policy network; alpha_lr: learning rate of the variable alpha
        soft_q_optimizer1 = tf.optimizers.Adam(soft_q_lr)
        soft_q_optimizer2 = tf.optimizers.Adam(soft_q_lr)
        policy_optimizer = tf.optimizers.Adam(policy_lr)
        alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        optimizers_list = [soft_q_optimizer1, soft_q_optimizer2, policy_optimizer, alpha_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=150,
        batch_size=64,
        explore_steps=200,
        update_itr=3,
        policy_target_update_interval=3,
        reward_scale=1.,
        AUTO_ENTROPY=True,
        train_episodes=100,
        test_episodes=10,
        save_interval=10,
    )

    return alg_params, learn_params


def mujoco(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        replay_buffer_capacity=5e5,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                     output_activation=None,
                                                     state_conditioned=True)
        net_list = [soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2, policy_net]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        soft_q_lr, policy_lr, alpha_lr = 3e-4, 3e-4, 3e-4  # soft_q_lr: learning rate of the Q network; policy_lr: learning rate of the policy network; alpha_lr: learning rate of the variable alpha
        soft_q_optimizer1 = tf.optimizers.Adam(soft_q_lr)
        soft_q_optimizer2 = tf.optimizers.Adam(soft_q_lr)
        policy_optimizer = tf.optimizers.Adam(policy_lr)
        alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        optimizers_list = [soft_q_optimizer1, soft_q_optimizer2, policy_optimizer, alpha_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=150,
        batch_size=64,
        explore_steps=200,
        update_itr=3,
        policy_target_update_interval=3,
        reward_scale=1.,
        AUTO_ENTROPY=True,
        train_episodes=100,
        test_episodes=10,
        save_interval=10,
    )

    return alg_params, learn_params


def robotics(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        replay_buffer_capacity=5e5,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                     output_activation=None,
                                                     state_conditioned=True)
        net_list = [soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2, policy_net]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        soft_q_lr, policy_lr, alpha_lr = 3e-4, 3e-4, 3e-4  # soft_q_lr: learning rate of the Q network; policy_lr: learning rate of the policy network; alpha_lr: learning rate of the variable alpha
        soft_q_optimizer1 = tf.optimizers.Adam(soft_q_lr)
        soft_q_optimizer2 = tf.optimizers.Adam(soft_q_lr)
        policy_optimizer = tf.optimizers.Adam(policy_lr)
        alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        optimizers_list = [soft_q_optimizer1, soft_q_optimizer2, policy_optimizer, alpha_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=150,
        batch_size=64,
        explore_steps=200,
        update_itr=3,
        policy_target_update_interval=3,
        reward_scale=1.,
        AUTO_ENTROPY=True,
        train_episodes=100,
        test_episodes=10,
        save_interval=10,
    )

    return alg_params, learn_params


def dm_control(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        replay_buffer_capacity=5e5,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                     output_activation=None,
                                                     state_conditioned=True)
        net_list = [soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2, policy_net]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        soft_q_lr, policy_lr, alpha_lr = 3e-4, 3e-4, 3e-4  # soft_q_lr: learning rate of the Q network; policy_lr: learning rate of the policy network; alpha_lr: learning rate of the variable alpha
        soft_q_optimizer1 = tf.optimizers.Adam(soft_q_lr)
        soft_q_optimizer2 = tf.optimizers.Adam(soft_q_lr)
        policy_optimizer = tf.optimizers.Adam(policy_lr)
        alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        optimizers_list = [soft_q_optimizer1, soft_q_optimizer2, policy_optimizer, alpha_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=150,
        batch_size=64,
        explore_steps=200,
        update_itr=3,
        policy_target_update_interval=3,
        reward_scale=1.,
        AUTO_ENTROPY=True,
        train_episodes=100,
        test_episodes=10,
        save_interval=10,
    )

    return alg_params, learn_params


def rlbench(env, default_seed=True):
    if default_seed:
        seed = 2
        set_seed(seed, env)  # reproducible

    alg_params = dict(
        replay_buffer_capacity=5e5,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                       hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = QNetwork(env.observation_space, env.action_space,
                                              hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                     output_activation=None,
                                                     state_conditioned=True)
        net_list = [soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2, policy_net]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        soft_q_lr, policy_lr, alpha_lr = 3e-4, 3e-4, 3e-4  # soft_q_lr: learning rate of the Q network; policy_lr: learning rate of the policy network; alpha_lr: learning rate of the variable alpha
        soft_q_optimizer1 = tf.optimizers.Adam(soft_q_lr)
        soft_q_optimizer2 = tf.optimizers.Adam(soft_q_lr)
        policy_optimizer = tf.optimizers.Adam(policy_lr)
        alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        optimizers_list = [soft_q_optimizer1, soft_q_optimizer2, policy_optimizer, alpha_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=150,
        batch_size=64,
        explore_steps=200,
        update_itr=3,
        policy_target_update_interval=3,
        reward_scale=1.,
        AUTO_ENTROPY=True,
        train_episodes=100,
        test_episodes=10,
        save_interval=10,
    )

    return alg_params, learn_params
