"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import operator
import os
import random
import copy

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense, Input
from tensorlayer.models import Model
from rlzoo.common.basic_nets import *
from rlzoo.common.distributions import make_dist
from gym import spaces

tfd = tfp.distributions
Normal = tfd.Normal


class StochasticContinuousPolicyNetwork(Model):
    def __init__(self, state_shape, action_shape, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
                 activation=tf.nn.relu, output_activation=None, log_std_min=-20, log_std_max=2, trainable=True):
        """ Stochastic continuous policy network with multiple fully-connected layers or convolutional layers (according to state shape)

        Args:
            state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
            action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            log_std_min (float): lower bound of standard deviation of action
            log_std_max (float): upper bound of standard deviation of action
            trainable (bool): set training and evaluation mode
        """

        action_dim = action_shape[0]
        if len(state_shape) == 1:
            with tf.name_scope('MLP'):
                state_dim = state_shape[0]
                inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation)
        else:
            with tf.name_scope('CNN'):
                inputs, l = CNN(state_shape, conv_kwargs=None)
        with tf.name_scope('Output_Mean'):
            mean_linear = Dense(n_units=action_dim, act=output_activation, W_init=w_init)(l)
        with tf.name_scope('Output_Std'):
            log_std_linear = Dense(n_units=action_dim, act=output_activation, W_init=w_init)(l)
            log_std_linear = tl.layers.Lambda(lambda x: tf.clip_by_value(x, log_std_min, log_std_max), name='Lambda')(
                log_std_linear)

        super().__init__(inputs=inputs, outputs=[mean_linear, log_std_linear])
        if trainable:
            self.train()
        else:
            self.eval()


class DeterministicContinuousPolicyNetwork(Model):
    def __init__(self, state_shape, action_shape, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(), \
                 activation=tf.nn.relu, output_activation=tf.nn.tanh, trainable=True):
        """ Deterministic continuous policy network with multiple fully-connected layers or convolutional layers (according to state shape)
        
        Args:
            state_shape (tuple[int]): shape of the state, for example, (state_dim, ) for single-dimensional state
            action_shape (tuple[int]): shape of the action, for example, (action_dim, ) for single-dimensional action
            hidden_dim_list (list[int]): a list of dimensions of hidden layers
            w_init (callable): weights initialization
            activation (callable): activation function
            output_activation (callable or None): output activation function
            trainable (bool): set training and evaluation mode
        """

        action_dim = action_shape[0]

        if len(state_shape) == 1:
            with tf.name_scope('MLP'):
                state_dim = state_shape[0]
                inputs, l = MLP(state_dim, hidden_dim_list, w_init, activation)
        else:
            with tf.name_scope('CNN'):
                inputs, l = CNN(state_shape, conv_kwargs=None)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=action_dim, act=output_activation, W_init=w_init)(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()


class DeterministicPolicyNetwork(Model):
    def __init__(self, state_space, action_space, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(),
                 activation=tf.nn.relu, output_activation=tf.nn.tanh, trainable=True, name=None):
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
        self._state_space, self._action_space = state_space, action_space

        if isinstance(self._action_space, spaces.Discrete):
            self._action_shape = self._action_space.n,

        elif isinstance(self._action_space, spaces.Box):
            assert len(self._action_space.shape) == 1
            self._action_shape = self._action_space.shape

            assert all(self._action_space.low < self._action_space.high)
            action_bounds = [self._action_space.low, self._action_space.high]
            self._action_mean = np.mean(action_bounds, 0)
            self._action_scale = action_bounds[1] - self._action_mean
        else:
            raise NotImplementedError

        self._state_shape = state_space.shape

        # build structure
        if len(self._state_shape) == 1:
            l = inputs = Input((None,) + self._state_shape, name='input_layer')
        else:
            with tf.name_scope('CNN'):
                inputs, l = CNN(self._state_shape, conv_kwargs=None)

        with tf.name_scope('MLP'):
            for i, dim in enumerate(hidden_dim_list):
                l = Dense(n_units=dim, act=activation, W_init=w_init, name='hidden_layer%d' % (i + 1))(l)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=self._action_shape[0], act=output_activation, W_init=w_init)(l)

            if isinstance(self._action_space, spaces.Discrete):
                outputs = tl.layers.Lambda(lambda x: tf.argmax(tf.nn.softmax(x), axis=-1))(outputs)
            elif isinstance(self._action_space, spaces.Box):
                outputs = tl.layers.Lambda(lambda x: x * self._action_scale + self._action_mean)(outputs)
                outputs = tl.layers.Lambda(lambda x: tf.clip_by_value(x, self._action_space.low,
                                                                      self._action_space.high))(outputs)

        # make model
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        if trainable:
            self.train()
        else:
            self.eval()

    def __call__(self, states, *args, **kwargs):
        if np.shape(states)[1:] != self.state_shape:
            raise ValueError(
                'Input state shape error. shape can be {} but your shape is {}'.format((None,) + self.state_shape,
                                                                                       np.shape(states)))
        states = np.array(states, dtype=np.float32)
        return super().__call__(states, *args, **kwargs)

    def random_sample(self):
        ''' generate random actions for exploration '''

        if isinstance(self._action_space, spaces.Discrete):
            return np.random.choice(self._action_space.n, 1)[0]
        else:
            return np.random.uniform(self._action_space.low, self._action_space.high, self._action_shape)

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


class StochasticPolicyNetwork(Model):
    def __init__(self, state_space, action_space, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(),
                 activation=tf.nn.relu, output_activation=tf.nn.tanh, log_std_min=-20, log_std_max=2, trainable=True,
                 name=None, state_conditioned=False):
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

        Tips:
            We recommend to use tf.nn.tanh for output_activation, especially for continuous action space,
            to ensure the final action range is exactly the same as declared in action space after action normalization.
        
        """
        self._state_space, self._action_space = state_space, action_space

        if isinstance(self._action_space, spaces.Discrete):
            self._action_shape = self._action_space.n,
            self.policy_dist = make_dist(self._action_space)  # create action distribution
        elif isinstance(self._action_space, spaces.Box):  # normalize action
            assert len(self._action_space.shape) == 1
            self._action_shape = self._action_space.shape

            assert all(self._action_space.low < self._action_space.high)
            action_bounds = [self._action_space.low, self._action_space.high]
            self._action_mean = np.mean(action_bounds, 0)
            self._action_scale = action_bounds[1] - self._action_mean

            self.policy_dist = make_dist(self._action_space)  # create action distribution
            self.policy_dist.action_mean = self._action_mean
            self.policy_dist.action_scale = self._action_scale
        else:
            raise NotImplementedError

        self._state_shape = state_space.shape
        self._state_conditioned = state_conditioned

        # build structure
        if len(self._state_shape) == 1:
            l = inputs = Input((None,) + self._state_shape, name='input_layer')
        else:
            with tf.name_scope('CNN'):
                inputs, l = CNN(self._state_shape, conv_kwargs=None)

        with tf.name_scope('MLP'):
            for i, dim in enumerate(hidden_dim_list):
                l = Dense(n_units=dim, act=activation, W_init=w_init, name='hidden_layer%d' % (i + 1))(l)

        with tf.name_scope('Output'):
            if isinstance(action_space, spaces.Discrete):
                outputs = Dense(n_units=self.policy_dist.ndim, act=output_activation, W_init=w_init)(l)
            elif isinstance(action_space, spaces.Box):
                mu = Dense(n_units=self.policy_dist.ndim, act=output_activation, W_init=w_init)(l)

                if self._state_conditioned:
                    log_sigma = Dense(n_units=self.policy_dist.ndim, act=None, W_init=w_init)(l)
                    log_sigma = tl.layers.Lambda(lambda x: tf.clip_by_value(x, log_std_min, log_std_max))(log_sigma)
                    outputs = [mu, log_sigma]
                else:
                    outputs = mu
                    self._log_sigma = tf.Variable(np.zeros(self.policy_dist.ndim, dtype=np.float32))
            else:
                raise NotImplementedError

        # make model
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        if isinstance(self._action_space, spaces.Box) and not self._state_conditioned:
            self.trainable_weights.append(self._log_sigma)

        if trainable:
            self.train()
        else:
            self.eval()

    def __call__(self, states, *args, greedy=False, **kwargs):
        if np.shape(states)[1:] != self.state_shape:
            raise ValueError(
                'Input state shape error. Shape should be {} but your shape is {}'.format((None,) + self.state_shape,
                                                                                          np.shape(states)))
        states = np.array(states, dtype=np.float32)
        params = super().__call__(states, *args, **kwargs)
        if isinstance(self._action_space, spaces.Box) and not self._state_conditioned:
            params = params, self._log_sigma
        self.policy_dist.set_param(params)
        if greedy:
            result = self.policy_dist.greedy_sample()
        else:
            result = self.policy_dist.sample()

        if isinstance(self._action_space, spaces.Box):  # normalize action
            if greedy:
                result = result * self._action_scale + self._action_mean
            else:
                result, explore = result
                result = result * self._action_scale + self._action_mean + explore

            result = tf.clip_by_value(result, self._action_space.low, self._action_space.high)
        return result

    def random_sample(self):
        ''' generate random actions for exploration '''

        if isinstance(self._action_space, spaces.Discrete):
            return np.random.choice(self._action_space.n, 1)[0]
        else:
            return np.random.uniform(self._action_space.low, self._action_space.high, self._action_shape)

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
