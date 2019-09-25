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

class MlpValueNetwork(Model):
    def __init__(self, state_dim, hidden_dim_list, name='', \
        weights_initialization='Glorot Normal', activation = 'Relu', trainable = True):
        
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
        outputs = Dense(n_units=1, act=act, W_init=w_init, name=name+'_output_layer')(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()    

class MlpQNetwork(Model):
    def __init__(self, action_dim, state_dim, hidden_dim_list, name='', \
        weights_initialization='Glorot Normal', activation = 'Relu', trainable = True):
        input_dim = action_dim + state_dim
        if weights_initialization == 'Glorot Normal' or weights_initialization == None:  # glorot normal as default
            w_init = tf.keras.initializers.glorot_normal(
                seed=None
            )
        # add other options
        
        if activation == 'Relu' or activation == None:  # relu as default
            act= tf.nn.relu
        # add other options
        
        l = inputs = Input([None, input_dim], name=name+'_input_layer')
        for i in range(len(hidden_dim_list)):
            suffix = '_hidden_layer%d' % (i+1)
            l = Dense(n_units=hidden_dim_list[i], act=act, W_init=w_init, name=name+suffix)(l)
        outputs = Dense(n_units=1, act=act, W_init=w_init, name=name+'_output_layer')(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()    