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

class ValueNetwork(Model):
    def __init__(self, state_shape, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
        activation = tf.nn.relu, output_activation = None, trainable = True):
        """ Value network with multiple fully-connected layers or convolutional layers (according to state shape)
        
        Args:
            state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            trainable (bool): set training and evaluation mode
        """

        if len(state_shape)==1:
            with tf.name_scope('MLP'):
                state_dim = state_shape[0]
                inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation)
        else:
            with tf.name_scope('CNN'):
                inputs, l = CNN(state_shape, conv_kwargs=None)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()    

class CnnBackbone(Model):
    def __init__(self, state_shape, conv_kwargs=None ):
        """ CNN feature extractor for high-dimensional state """
        inputs, outputs = CNN(state_shape, conv_kwargs)
        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()

class MlpQNetwork(Model):
    def __init__(self, state_shape, action_shape, hidden_dim_list, \
        w_init=tf.keras.initializers.glorot_normal(), activation = tf.nn.relu, output_activation = None, trainable = True):
        """ Q-value network with multiple fully-connected layers 

        Inputs: (state tensor, action tensor)

        Args:
            state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
            action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            trainable (bool): set training and evaluation mode
        """

        input_shape = tuple(map(sum,zip(action_shape,state_shape)))
        input_dim = input_shape[0]

        assert len(state_shape)==1
        with tf.name_scope('MLP'):
            inputs, l = MLP(input_dim, hidden_dim_list, w_init, activation)
        # need to handle image state with action for q here, cnn for state first

        with tf.name_scope('Output'):
            outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()    



class QNetwork(Model):
    def __init__(self, state_shape, action_shape, hidden_dim_list, \
        w_init=tf.keras.initializers.glorot_normal(), activation = tf.nn.relu, output_activation = None, trainable = True):
        """ Q-value network with multiple fully-connected layers or convolutional layers (according to state shape)

        Inputs: (state tensor, action tensor)

        Args:
            state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
            action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            trainable (bool): set training and evaluation mode
        """
        self.name = 'Q_Network'
        if len(state_shape) == 1:
            input_shape = tuple(map(sum,zip(action_shape,state_shape)))
            input_dim = input_shape[0]
            self.mlp = MLP(input_dim, hidden_dim_list, w_init, activation)
        elif len(state_shape) >1:
            self.cnn = CNN(state_shape, conv_kwargs=None)
        else:
            raise ValueError("State Shape Not Accepted!")

        assert len(action_shape) == 1

    def forward(self, s, a):
        if self.state_shape == 1:
            with tf.name_scope('MLP'):
                sa=tf.concat([s,a],axis=-1)
                _, l = self.mlp(sa)

        else:  # high-dimensional state
            with tf.name_scope('CNN'):
                _, l = self.cnn(s)  # extracted state feature
            l=tf.concat([l, a], axis=-1)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(l)

        return outputs
