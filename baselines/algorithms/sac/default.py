import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *
from common.distributions import make_dist
from gym import spaces



def atari(env):
    state_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)

    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        gamma = 0.9,
        replay_buffer_capacity = 5e5,
        action_range = 1.
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4 #number of hidden layers for the networks
        hidden_dim=64 # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
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
        policy_target_update_interval = 3,  
        reward_scale = 1. , 
        seed=2, 
        AUTO_ENTROPY = True, 
        DETERMINISTIC = False
    )

    return alg_params, learn_params


def classic_control(env):
    action_shape = env.action_space.shape  # only continuous
    state_shape = env.observation_space.shape


    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        replay_buffer_capacity = 5e5,
        action_range = 1.
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4 #number of hidden layers for the networks
        hidden_dim=64 # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
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
        policy_target_update_interval = 3,  
        reward_scale = 1. , 
        seed=2, 
        AUTO_ENTROPY = True, 
        DETERMINISTIC = False
    )

    return alg_params, learn_params


def rlbench(env):
    action_shape = env.action_space.shape  # only continuous
    state_shape = env.observation_space.shape


    alg_params = dict(
        state_dim = state_shape[0],
        action_dim = action_shape[0],
        replay_buffer_capacity = 5e5,
        action_range = 0.1
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4 #number of hidden layers for the networks
        hidden_dim=64 # dimension of hidden layers for the networks, default as the same for each layer here
        with tf.name_scope('SAC'):
            with tf.name_scope('Q_Net1'):
                soft_q_net1 = QNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Q_Net2'):
                soft_q_net2 = QNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Target_Q_Net1'):
                target_soft_q_net1 = QNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Target_Q_Net2'):
                target_soft_q_net2 = QNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
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
        policy_target_update_interval = 3,  
        reward_scale = 1. , 
        seed=2, 
        AUTO_ENTROPY = True, 
        DETERMINISTIC = False
    )

    return alg_params, learn_params
