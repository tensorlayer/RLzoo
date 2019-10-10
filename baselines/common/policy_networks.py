"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import operator
import os
import random

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense, Input
from tensorlayer.models import Model
from common.basic_nets import * 
from common.distributions import make_dist

tfd = tfp.distributions
Normal = tfd.Normal

# class StochasticPolicyNetwork(Model):
#     def __init__(self, state_shape, action_shape, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
#         activation = tf.nn.relu, output_activation = None, log_std_min=-20, log_std_max=2, trainable=True):
#         """ Stochastic continuous policy network with multiple fully-connected layers 
        
#         Args:
#             state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
#             action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
#             hidden_dim_list (list[int]): a list of dimensions of hidden layers
#             w_init (callable): weights initialization
#             activation (callable): activation function
#             output_activation (callable or None): output activation function
#             log_std_min (float): lower bound of standard deviation of action
#             log_std_max (float): upper bound of standard deviation of action
#             trainable (bool): set training and evaluation mode
#         """

#         action_dim = action_shape[0]
#         state_dim = state_shape[0]  # need modification for cnn
#         with tf.name_scope('MLP'):
#             inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation)
#         with tf.name_scope('Output_Mean'):
#             mean_linear = Dense(n_units=action_dim, act=output_activation, W_init=w_init)(l)
#         with tf.name_scope('Output_Std'):
#             log_std_linear = Dense(n_units=action_dim, act=output_activation, W_init=w_init)(l)  
#             log_std_linear = tl.layers.Lambda(lambda x: tf.clip_by_value(x, log_std_min, log_std_max), name='Lambda')(log_std_linear)

#         super().__init__(inputs=inputs, outputs=[mean_linear, log_std_linear])
#         if trainable:
#             self.train()
#         else:
#             self.eval()    

# class DeterministicPolicyNetwork(Model):
#     def __init__(self, state_shape, action_shape, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
#         activation = tf.nn.relu, output_activation = tf.nn.tanh, trainable = True):
#         """ Deterministic continuous policy network with multiple fully-connected layers 
        
#         Args:
#             state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
#             action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
#             hidden_dim_list (list[int]): a list of dimensions of hidden layers
#             w_init (callable): weights initialization
#             activation (callable): activation function
#             output_activation (callable or None): output activation function
#             trainable (bool): set training and evaluation mode
#         """

#         action_dim = action_shape[0]
#         state_dim = state_shape[0]
#         with tf.name_scope('MLP'):
#             inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation)
#         with tf.name_scope('Output'):
#             outputs = Dense(n_units=action_dim, act=output_activation, W_init=w_init)(l)

#         super().__init__(inputs=inputs, outputs=outputs)
#         if trainable:
#             self.train()
#         else:
#             self.eval()   

class DeterministicPolicyNetwork(Model):
    def __init__(self, state_space, action_space, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
        activation = tf.nn.relu, output_activation = tf.nn.tanh, trainable = True):
        """ Deterministic continuous/discrete policy network with multiple fully-connected layers 
        
        Args:
            state_space (gym.spaces): space of the state from gym environments
            action_space (gym.spaces): space of the action from gym environments
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            trainable (bool): set training and evaluation mode
        """
        self.policy_dist = make_dist(action_space)

        state_shape = state_space.shape
        state_dim = state_shape[0]  # need modification for cnn
        with tf.name_scope('MLP'):
            inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation)
        with tf.name_scope('Output'):
            outputs = Dense(n_units=self.policy_dist.output_dim, act=output_activation, W_init=w_init)(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()   

class StochasticPolicyNetwork(Model):
    def __init__(self, state_space, action_space, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
        activation = tf.nn.relu, output_activation = None, log_std_min=-20, log_std_max=2, trainable=True):
        """ Stochastic continuous/discrete policy network with multiple fully-connected layers 
        
        Args:
            state_space (gym.spaces): space of the state from gym environments
            action_space (gym.spaces): space of the action from gym environments
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            log_std_min (float): lower bound of standard deviation of action
            log_std_max (float): upper bound of standard deviation of action
            trainable (bool): set training and evaluation mode
        """
        self.policy_dist = make_dist(action_space)

        state_shape = state_space.shape
        state_dim = state_shape[0]  # need modification for cnn
        with tf.name_scope('MLP'):
            inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=self.policy_dist.output_dim, act=output_activation, W_init=w_init)(l)
        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()   
