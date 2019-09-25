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


tfd = tfp.distributions
Normal = tfd.Normal

class StochasticPolicyNetwork(Model):
    ''' stochastic continuous policy network for generating action according to the state '''

    def __init__(self, state_dim, action_dim, hidden_dim_list, weights_initialization='Glorot Normal', \
         activation = 'Relu', log_std_min=-20, log_std_max=2, name='', trainable=True):
        if weights_initialization == 'Glorot Normal' or weights_initialization == None:  # glorot normal as default
            w_init = tf.keras.initializers.glorot_normal(
                seed=None
            )
        # add other options
        
        if activation == 'Relu' or activation == None:  # relu as default
            act= tf.nn.relu
        # add other options

        l = inputs = Input([None, state_dim], name=name+'_input_layer')
        for i in range(len(hidden_dim_list)):
            suffix = '_hidden_layer%d' % (i+1)
            l = Dense(n_units=hidden_dim_list[i], act=act, W_init=w_init, name=name+suffix)(l)
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
    ''' deterministic continuous policy network for generating action according to the state '''

    def __init__(self, state_dim, action_dim, hidden_dim_list, weights_initialization='Random Uniform', \
         activation = 'Relu', log_std_min=-20, log_std_max=2, name='', trainable = True):
        if weights_initialization == 'Glorot Normal' or weights_initialization == None:  # glorot normal as default
            w_init = tf.keras.initializers.glorot_normal(
                seed=None
            )
        elif weights_initialization== 'Random Uniform':
            w_init=tf.random_uniform_initializer(-3e-3, 3e-3)
        # add other options
        
        if activation == 'Relu' or activation == None:  # relu as default
            act= tf.nn.relu
        # add other options

        l = inputs = Input([None, state_dim], name=name+'_input_layer')
        for i in range(len(hidden_dim_list)):
            suffix = '_hidden_layer%d' % (i+1)
            l = Dense(n_units=hidden_dim_list[i], act=act, W_init=w_init, name=name+suffix)(l)
        outputs = Dense(n_units=action_dim, W_init=w_init, name=name+'_output_layer')(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()   
