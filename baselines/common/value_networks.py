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
import copy

import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense, Input
from tensorlayer.models import Model
from common.basic_nets import * 
from gym import spaces

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

        with tf.name_scope('Output'):
            outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()    



# class QNetwork(Model):
#     def __init__(self, state_shape, action_shape, hidden_dim_list, \
#         w_init=tf.keras.initializers.glorot_normal(), activation = tf.nn.relu, output_activation = None, trainable = True):
#         """ Q-value network with multiple fully-connected layers or convolutional layers (according to state shape)

#         Args:
#             state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
#             action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
#             hidden_dim_list (list[int]): a list of dimensions of hidden layers
#             w_init (callable): weights initialization
#             activation (callable): activation function
#             output_activation (callable or None): output activation function
#             trainable (bool): set training and evaluation mode
#         """
#         super(QNetwork, self).__init__()
#         if len(state_shape) == 1:
#             input_shape = tuple(map(sum,zip(action_shape,state_shape)))
#             input_dim = input_shape[0]
#             self.mlp = MLPModel(input_dim, hidden_dim_list, w_init, activation)
#             self.output = Dense(n_units=1, act=output_activation, W_init=w_init, in_channels=hidden_dim_list[-1])
#         elif len(state_shape) >1:
#             self.cnn = CNNModel(state_shape, conv_kwargs=None)
#             self.output = Dense(n_units=1, act=output_activation, W_init=w_init, in_channels=9224)  # 9216+8
#         else:
#             raise ValueError("State Shape Not Accepted!")
        
#         assert len(action_shape) == 1
#         self.state_shape = state_shape
#         self.output_activation = output_activation
#         self.w_init = w_init

#         if trainable:
#             self.train()
#         else:
#             self.eval()  
  

#     def forward(self, inputs):
#         """ Inputs: (state tensor, action tensor) """
#         [s,a] = inputs
#         if len(self.state_shape) == 1:
#             with tf.name_scope('MLP'):
#                 sa=tf.concat([s,a],axis=-1)
#                 l = self.mlp(sa)

#         else:  # high-dimensional state
#             with tf.name_scope('CNN'):
#                 l = self.cnn(s)  # extracted state feature
#             l = tf.concat([l, a], axis=-1)

#         with tf.name_scope('Output'):
#             outputs = self.output(l)
#         return outputs


class QNetwork(Model):
    def __init__(self, state_space, action_space, hidden_dim_list,
                 w_init=tf.keras.initializers.glorot_normal(), activation=tf.nn.relu, output_activation=None,
                 trainable=True):
        """ Q-value network with multiple fully-connected layers or convolutional layers (according to state shape)

        Args:
            state_space (gym.spaces): space of the state from gym environments
            action_space (gym.spaces): space of the action from gym environments
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            trainable (bool): set training and evaluation mode
        """
        state_shape = state_space.shape

        self._state_space, self._action_space = state_space, action_space

        if isinstance(action_space, spaces.Discrete):
            action_shape = action_space.n,
            act_inputs = Input((None,), name='Act_Input_Layer', dtype=tf.int32)
        elif isinstance(action_space, spaces.Box):
            action_shape = action_space.shape
            assert len(action_shape) == 1
            act_inputs = Input((None,) + action_shape, name='Act_Input_Layer')
        else:
            raise NotImplementedError

        if len(state_shape) == 1:
            obs_inputs = current_layer = Input((None,) + state_shape, name='Obs_Input_Layer')
        elif len(state_shape) > 1:
            with tf.name_scope('QNet_CNN'):
                obs_inputs, current_layer = CNN(state_shape)
        else:
            raise ValueError("State Shape Not Accepted!")

        if isinstance(action_space, spaces.Box):
            current_layer = tl.layers.Concat(-1)([current_layer, act_inputs])

        with tf.name_scope('QNet_MLP'):
            for i, dim in enumerate(hidden_dim_list):
                current_layer = Dense(n_units=dim, act=activation, W_init=w_init,
                                      name='mlp_hidden_layer%d' % (i + 1))(current_layer)

            if isinstance(action_space, spaces.Discrete):
                current_layer = Dense(n_units=action_shape[0], act=output_activation, W_init=w_init)(current_layer)

                act_one_hot = tl.layers.OneHot(depth=action_shape[0], axis=1)(act_inputs)
                outputs = tl.layers.Lambda(
                    lambda x: tf.reduce_sum(tf.reduce_prod(x, axis=0), axis=1))((current_layer, act_one_hot))
            elif isinstance(action_space, spaces.Box):
                outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(current_layer)
            else:
                raise ValueError("State Shape Not Accepted!")

        super().__init__(inputs=[obs_inputs, act_inputs], outputs=outputs)

        if trainable:
            self.train()
        else:
            self.eval()

    @property
    def state_space(self):
        return copy.deepcopy(self._state_space)

    @property
    def action_space(self):
        return copy.deepcopy(self._action_space)
