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
from rlzoo.common.basic_nets import *
from gym import spaces

tfd = tfp.distributions
Normal = tfd.Normal


class ValueNetwork(Model):
    def __init__(self, state_space, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(),
                 activation=tf.nn.relu, output_activation=None, trainable=True, name=None):
        """ Value network with multiple fully-connected layers or convolutional layers (according to state shape)
        
        Args:
            state_space (gym.spaces): space of the state from gym environments
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            trainable (bool): set training and evaluation mode
        """
        self._state_space = state_space

        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            with tf.name_scope('MLP'):
                inputs, l = MLP(self._state_shape[0], hidden_dim_list, w_init, activation)
        else:
            with tf.name_scope('CNN'):
                inputs, l = CNN(self._state_shape, conv_kwargs=None)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(l)

        super().__init__(inputs=inputs, outputs=outputs, name=name)
        if trainable:
            self.train()
        else:
            self.eval()

    def __call__(self, states, *args, **kwargs):
        if np.shape(states)[1:] != self.state_shape:
            raise ValueError(
                'Input state shape error. Shape can be {} but your shape is {}'.format((None,) + self.state_shape,
                                                                                       np.shape(states)))
        states = np.array(states, dtype=np.float32)
        return super().__call__(states, *args, **kwargs)

    @property
    def state_space(self):
        return copy.deepcopy(self._state_space)

    @property
    def state_shape(self):
        return copy.deepcopy(self._state_shape)


class MlpQNetwork(Model):
    def __init__(self, state_shape, action_shape, hidden_dim_list, \
                 w_init=tf.keras.initializers.glorot_normal(), activation=tf.nn.relu, output_activation=None,
                 trainable=True):
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

        input_shape = tuple(map(sum, zip(action_shape, state_shape)))
        input_dim = input_shape[0]

        assert len(state_shape) == 1
        with tf.name_scope('MLP'):
            inputs, l = MLP(input_dim, hidden_dim_list, w_init, activation)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()


class QNetwork(Model):
    def __init__(self, state_space, action_space, hidden_dim_list,
                 w_init=tf.keras.initializers.glorot_normal(), activation=tf.nn.relu, output_activation=None,
                 trainable=True, name=None):
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
        self._state_space, self._action_space = state_space, action_space

        self._state_shape = state_space.shape

        if isinstance(self._action_space, spaces.Discrete):
            self._action_shape = self._action_space.n,
            act_inputs = Input((None,), name='Act_Input_Layer', dtype=tf.int32)
        elif isinstance(self._action_space, spaces.Box):
            self._action_shape = self._action_space.shape
            assert len(self._action_shape) == 1
            act_inputs = Input((None,) + self._action_shape, name='Act_Input_Layer')
        else:
            raise NotImplementedError

        if len(self._state_shape) == 1:
            obs_inputs = current_layer = Input((None,) + self._state_shape, name='Obs_Input_Layer')
        elif len(self._state_shape) > 1:
            with tf.name_scope('QNet_CNN'):
                obs_inputs, current_layer = CNN(self._state_shape)
        else:
            raise ValueError("State Shape Not Accepted!")

        if isinstance(self._action_space, spaces.Box):
            current_layer = tl.layers.Concat(-1)([current_layer, act_inputs])

        with tf.name_scope('QNet_MLP'):
            for i, dim in enumerate(hidden_dim_list):
                current_layer = Dense(n_units=dim, act=activation, W_init=w_init,
                                      name='mlp_hidden_layer%d' % (i + 1))(current_layer)

        with tf.name_scope('Outputs'):
            if isinstance(self._action_space, spaces.Discrete):
                current_layer = Dense(n_units=self._action_shape[0], act=output_activation, W_init=w_init)(
                    current_layer)

                act_one_hot = tl.layers.OneHot(depth=self._action_shape[0], axis=1)(act_inputs)  # discrete action choice to one-hot vector
                outputs = tl.layers.Lambda(
                    lambda x: tf.reduce_sum(tf.reduce_prod(x, axis=0), axis=1))((current_layer, act_one_hot))
            elif isinstance(self._action_space, spaces.Box):
                outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(current_layer)
            else:
                raise ValueError("State Shape Not Accepted!")

        super().__init__(inputs=[obs_inputs, act_inputs], outputs=outputs, name=name)

        if trainable:
            self.train()
        else:
            self.eval()

    def __call__(self, states_actions, *args, **kwargs):
        states, actions = states_actions
        if np.shape(states)[1:] != self.state_shape:
            raise ValueError(
                'Input state shape error. Shape can be {} but your shape is {}'.format((None,) + self.state_shape,
                                                                                       np.shape(states)))
        if len(states) != len(actions):
            raise ValueError(
                'Length of states and actions not match. States length is {} but actions length is {}'.format(
                    len(states),
                    len(actions)))

        if isinstance(self._action_space, spaces.Discrete) and np.any(actions % 1):
            raise ValueError('Input float actions in discrete action space')

        states = np.array(states, dtype=np.float32)
        # if isinstance(self._action_space, spaces.Discrete) and type(actions) == tf.int32:
        if isinstance(self._action_space, spaces.Discrete):
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        # elif isinstance(self._action_space, spaces.Box) and type(actions) == tf.float32:
        elif isinstance(self._action_space, spaces.Box):
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        return super().__call__([states, actions], *args, **kwargs)

    @property
    def state_space(self):
        return copy.deepcopy(self._state_space)

    @property
    def action_space(self):
        return copy.deepcopy(self._action_space)

    @property
    def state_shape(self):
        return copy.deepcopy(self._state_shape)

    @property
    def action_shape(self):
        return copy.deepcopy(self._action_shape)

