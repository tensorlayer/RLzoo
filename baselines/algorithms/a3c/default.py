import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *


def atari(env):
    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict()
    
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4 #number of hidden layers for the networks
        hidden_dim = 64 # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env+1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                        critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
                with tf.name_scope('Actor'):
                        actor = StochasticPolicyNetwork(env.observation_space, env.action_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list=[a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2, 
        max_steps=1000,
        gamma = 0.9
    )

    return alg_params, learn_params


def classic_control(env):
    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict()
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4 #number of hidden layers for the networks
        hidden_dim = 64 # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env+1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                        critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
                with tf.name_scope('Actor'):
                        actor = StochasticPolicyNetwork(env.observation_space, env.action_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list=[a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2, 
        max_steps=100,
        gamma = 0.9
    )

    return alg_params, learn_params
