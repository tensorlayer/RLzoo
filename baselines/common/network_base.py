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

class MlpNetwork(Model):
    def __init__(self, input_dim, output_dim, hidden_dim_list, num_hidden_layer, name: str, \\
        weights_initialization='Glorot Normal', activation = 'Relu', trainable = True):
        
        if self.weights_initialization == 'Glorot_Normal' or self.weights_initialization == None:  # glorot normal as default
            w_init = tf.keras.initializers.glorot_normal(
                seed=None
            )
        # add other options
        
        if self.activation == 'Relu' or self.activation == None:  # relu as default
            act= tf.nn.relu
        # add other options
        
        l = inputs = Input([None, self.input_dim], name=self.name+'_input_layer')
        for i in range(self.num_hidden_layer):
            l = Dense(n_units=self.hidden_dim, act=act, W_init=w_init, name=self.name+'_hidden_layer%d'(i+1))(l)
        outputs = Dense(n_units=self.output_dim, act=act, W_init=w_init, name=self.name+'_output_layer')(l)
        return tl.models.Model(inputs=inputs, outputs=outputs, name='mlp_network')
          

class CnnNetwork(Model):
    def __init__(self, ):
        pass


class RnnNetwork(Model):
    def __init__(self, ):
        pass




