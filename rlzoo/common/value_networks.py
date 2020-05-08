"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import copy

import numpy as np
import tensorlayer as tl
from tensorlayer.layers import Dense, Input
from tensorlayer.models import Model

from rlzoo.common.basic_nets import *


class ValueNetwork(Model):
    def __init__(self, state_space, hidden_dim_list, w_init=tf.keras.initializers.glorot_normal(),
                 activation=tf.nn.relu, output_activation=None, trainable=True, name=None):
        """ 
        Value network with multiple fully-connected layers or convolutional layers (according to state shape)
        
        :param state_space: (gym.spaces) space of the state from gym environments
        :param hidden_dim_list: (list[int]) a list of dimensions of hidden layers
        :param w_init: (callable) weights initialization
        :param activation: (callable) activation function
        :param output_activation: (callable or None) output activation function
        :param trainable: (bool) set training and evaluation mode
        """
        self._state_space = state_space

        obs_inputs, current_layer, self._state_shape = CreateInputLayer(state_space)

        if isinstance(state_space, spaces.Dict):
            assert isinstance(obs_inputs, OrderedDict)
            assert isinstance(current_layer, OrderedDict)
            self.input_dict = obs_inputs
            obs_inputs = list(obs_inputs.values())
            current_layer = tl.layers.Concat(-1)(list(current_layer.values()))

        with tf.name_scope('MLP'):
            for i, dim in enumerate(hidden_dim_list):
                current_layer = Dense(n_units=dim, act=activation, W_init=w_init, name='hidden_layer%d' % (i + 1))(
                    current_layer)

        with tf.name_scope('Output'):
            outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(current_layer)

        super().__init__(inputs=obs_inputs, outputs=outputs, name=name)
        if trainable:
            self.train()
        else:
            self.eval()

    def __call__(self, states, *args, **kwargs):
        if isinstance(self._state_space, spaces.Dict):
            states = np.array(states).transpose([1, 0]).tolist()
        else:
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
        """ 
        Q-value network with multiple fully-connected layers 

        Inputs: (state tensor, action tensor)

        :param state_shape: (tuple[int]) shape of the state, for example, (state_dim, ) for single-dimensional state
        :param action_shape: (tuple[int]) shape of the action, for example, (action_dim, ) for single-dimensional action
        :param hidden_dim_list: (list[int]) a list of dimensions of hidden layers
        :param w_init: (callable) weights initialization
        :param activation: (callable) activation function
        :param output_activation: (callable or None) output activation function
        :param trainable: (bool) set training and evaluation mode
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
                 trainable=True, name=None, state_only=False, dueling=False):
        """ Q-value network with multiple fully-connected layers or convolutional layers (according to state shape)

        :param state_space: (gym.spaces) space of the state from gym environments
        :param action_space: (gym.spaces) space of the action from gym environments
        :param hidden_dim_list: (list[int]) a list of dimensions of hidden layers
        :param w_init: (callable) weights initialization
        :param activation: (callable) activation function
        :param output_activation: (callable or None) output activation function
        :param trainable: (bool) set training and evaluation mode
        :param name: (str) name the model
        :param state_only: (bool) only input state or not, available in discrete action space
        :param dueling: (bool) whether use the dueling output or not, available in discrete action space
        """
        self._state_space, self._action_space = state_space, action_space
        self.state_only = state_only
        self.dueling = dueling

        # create state input layer
        obs_inputs, current_layer, self._state_shape = CreateInputLayer(state_space)

        # create action input layer
        if isinstance(self._action_space, spaces.Discrete):
            self._action_shape = self._action_space.n,
            if not self.state_only:
                act_inputs = Input((None,), name='Act_Input_Layer', dtype=tf.int64)
        elif isinstance(self._action_space, spaces.Box):
            self._action_shape = self._action_space.shape
            assert len(self._action_shape) == 1
            act_inputs = Input((None,) + self._action_shape, name='Act_Input_Layer')
        else:
            raise NotImplementedError

        # concat multi-head state
        if isinstance(state_space, spaces.Dict):
            assert isinstance(obs_inputs, dict)
            assert isinstance(current_layer, dict)
            self.input_dict = obs_inputs
            obs_inputs = list(obs_inputs.values())
            current_layer = tl.layers.Concat(-1)(list(current_layer.values()))

        if isinstance(self._action_space, spaces.Box):
            current_layer = tl.layers.Concat(-1)([current_layer, act_inputs])

        with tf.name_scope('QNet_MLP'):
            for i, dim in enumerate(hidden_dim_list):
                current_layer = Dense(n_units=dim, act=activation, W_init=w_init,
                                      name='mlp_hidden_layer%d' % (i + 1))(current_layer)

        with tf.name_scope('Outputs'):
            if isinstance(self._action_space, spaces.Discrete):
                if self.dueling:
                    v = Dense(1, None, tf.initializers.Orthogonal(1.0))(current_layer)
                    q = Dense(n_units=self._action_shape[0], act=output_activation, W_init=w_init)(
                        current_layer)
                    mean_q = tl.layers.Lambda(lambda x: tf.reduce_mean(x, 1, True))(q)
                    current_layer = tl.layers.Lambda(lambda x: x[0] + x[1] - x[2])((v, q, mean_q))
                else:
                    current_layer = Dense(n_units=self._action_shape[0], act=output_activation, W_init=w_init)(
                        current_layer)

                if not self.state_only:
                    act_one_hot = tl.layers.OneHot(depth=self._action_shape[0], axis=1)(
                        act_inputs)  # discrete action choice to one-hot vector
                    outputs = tl.layers.Lambda(
                        lambda x: tf.reduce_sum(tf.reduce_prod(x, axis=0), axis=1))((current_layer, act_one_hot))
                else:
                    outputs = current_layer

            elif isinstance(self._action_space, spaces.Box):
                outputs = Dense(n_units=1, act=output_activation, W_init=w_init)(current_layer)
            else:
                raise ValueError("State Shape Not Accepted!")

        if isinstance(state_space, spaces.Dict):
            if self.state_only:
                super().__init__(inputs=obs_inputs, outputs=outputs, name=name)
            else:
                super().__init__(inputs=obs_inputs + [act_inputs], outputs=outputs, name=name)
        else:
            if self.state_only:
                super().__init__(inputs=obs_inputs, outputs=outputs, name=name)
            else:
                super().__init__(inputs=[obs_inputs, act_inputs], outputs=outputs, name=name)
        if trainable:
            self.train()
        else:
            self.eval()

    def __call__(self, inputs, *args, **kwargs):
        if self.state_only:
            states = inputs
        else:
            states, actions = inputs

        # states and actions must have the same length
        if not self.state_only and len(states) != len(actions):
            raise ValueError(
                'Length of states and actions not match. States length is {} but actions length is {}'.format(
                    len(states),
                    len(actions)))

        if isinstance(self._state_space, spaces.Dict):
            states = np.array(states).transpose([1, 0]).tolist()  # batch states to multi-head
            ssv = list(self._state_shape.values())
            # check state shape
            for i, each_head in enumerate(states):
                if np.shape(each_head)[1:] != ssv[i]:
                    raise ValueError('Input state shape error.')

        else:
            if np.shape(states)[1:] != self.state_shape:
                raise ValueError(
                    'Input state shape error. Shape can be {} but your shape is {}'.format((None,) + self.state_shape,
                                                                                           np.shape(states)))
            states = np.array(states, dtype=np.float32)

        if not self.state_only:
            if isinstance(self._action_space, spaces.Discrete) and np.any(actions % 1):
                raise ValueError('Input float actions in discrete action space')
            if isinstance(self._action_space, spaces.Discrete):
                actions = tf.convert_to_tensor(actions, dtype=tf.int64)
            elif isinstance(self._action_space, spaces.Box):
                actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            if isinstance(self._state_space, spaces.Dict):
                return super().__call__(states + [actions], *args, **kwargs)
            else:
                return super().__call__([states, actions], *args, **kwargs)
        else:
            return super().__call__(states, *args, **kwargs)


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

