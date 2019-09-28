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

tfd = tfp.distributions
Normal = tfd.Normal

class StochasticPolicyNetwork(Model):
    def __init__(self, state_shape, action_shape, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
        activation = tf.nn.relu, log_std_min=-20, log_std_max=2, name='', trainable=True):
        """ Stochastic continuous policy network with multiple fully-connected layers 
        
        Args:
            state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
            action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            log_std_min (float): lower bound of standard deviation of action
            log_std_max (float): upper bound of standard deviation of action
            name (str): name prefix
            trainable (bool): set training and evaluation mode
        """

        action_dim = action_shape[0]
        state_dim = state_shape[0]  # need modification for cnn

        inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation, name)
        mean_linear = Dense(n_units=action_dim, W_init=w_init, name=name+'_mean')(l)
        log_std_linear = Dense(n_units=action_dim, W_init=w_init, name=name+'_std')(l)  
        log_std_linear = tl.layers.Lambda(lambda x: tf.clip_by_value(x, log_std_min, log_std_max), 
        name=name+'_lambda')(log_std_linear)

        super().__init__(inputs=inputs, outputs=[mean_linear, log_std_linear])
        if trainable:
            self.train()
        else:
            self.eval()    

class DeterministicPolicyNetwork(Model):
    def __init__(self, state_shape, action_shape, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
        activation = tf.nn.relu, name='', trainable = True):
        """ Deterministic continuous policy network with multiple fully-connected layers 
        
        Args:
            state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
            action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            name (str): name prefix
            trainable (bool): set training and evaluation mode
        """

        action_dim = action_shape[0]
        state_dim = state_shape[0]
        
        inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation, name)
        outputs = Dense(n_units=action_dim, act=tf.nn.tanh, W_init=w_init, name=name+'_output_layer')(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()   

