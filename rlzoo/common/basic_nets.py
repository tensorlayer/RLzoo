"""Basic neural networks"""
import math

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense, Input


def MLP(input_dim, hidden_dim_list, w_init=tf.initializers.Orthogonal(0.2),
        activation=tf.nn.relu, *args, **kwargs):
    """Multiple fully-connected layers for approximation

    Args:
        input_dim (int): size of input tensor
        hidden_dim_list (list[int]): a list of dimensions of hidden layers
        w_init (callable): initialization method for weights
        activation (callable): activation function of hidden layers
    Return:
        input tensor, output tensor
    """

    l = inputs = Input([None, input_dim], name='input_layer')
    for i in range(len(hidden_dim_list)):
        l = Dense(n_units=hidden_dim_list[i], act=activation, W_init=w_init, name='mlp_layer%d' % (i + 1))(l)
    outputs = l

    return inputs, outputs


def MLPModel(input_dim, hidden_dim_list, w_init=tf.initializers.Orthogonal(0.2),
             activation=tf.nn.relu, *args, **kwargs):
    """Multiple fully-connected layers for approximation

    Args:
        input_dim (int): size of input tensor
        hidden_dim_list (list[int]): a list of dimensions of hidden layers
        w_init (callable): initialization method for weights
        activation (callable): activation function of hidden layers
    Return:
        input tensor, output tensor
    """
    l = inputs = Input([None, input_dim], name='Input_Layer')
    for i in range(len(hidden_dim_list)):
        l = Dense(n_units=hidden_dim_list[i], act=activation, W_init=w_init, name='Hidden_Layer%d' % (i + 1))(l)
    outputs = l

    return tl.models.Model(inputs=inputs, outputs=outputs)


def CNN(input_shape, conv_kwargs=None):
    """Multiple convolutional layers for approximation
    Default setting is equal to architecture used in DQN

    Args:
        input_shape (tuple[int]): (H, W, C)
        conv_kwargs (list[param]): list of conv parameters for tl.layers.Conv2d
    Return:
        input tensor, output tensor
    """
    if not conv_kwargs:
        in_channels = input_shape[-1]
        conv_kwargs = [
            {
                'in_channels': in_channels, 'n_filter': 32, 'act': tf.nn.relu,
                'filter_size': (8, 8), 'strides': (4, 4), 'padding': 'VALID',
                'W_init': tf.initializers.GlorotUniform()
            },
            {
                'in_channels': 32, 'n_filter': 64, 'act': tf.nn.relu,
                'filter_size': (4, 4), 'strides': (2, 2), 'padding': 'VALID',
                'W_init': tf.initializers.GlorotUniform()
            },
            {
                'in_channels': 64, 'n_filter': 64, 'act': tf.nn.relu,
                'filter_size': (3, 3), 'strides': (1, 1), 'padding': 'VALID',
                'W_init': tf.initializers.GlorotUniform()
            }
        ]
    l = inputs = tl.layers.Input((1,) + input_shape, name='input_layer')

    for i, kwargs in enumerate(conv_kwargs):
        kwargs['name'] = kwargs.get('name', 'cnn_layer{}'.format(i + 1))
        l = tl.layers.Conv2d(**kwargs)(l)
    outputs = tl.layers.Flatten(name='flatten_layer')(l)

    return inputs, outputs


def CNNModel(input_shape, conv_kwargs=None):
    """Multiple convolutional layers for approximation
    Default setting is equal to architecture used in DQN

    Args:
        input_shape (tuple[int]): (H, W, C)
        conv_kwargs (list[param]): list of conv parameters for tl.layers.Conv2d
    Return:
        tl.model.Model
    """
    if not conv_kwargs:
        in_channels = input_shape[-1]
        conv_kwargs = [
            {
                'in_channels': in_channels, 'n_filter': 32, 'act': tf.nn.relu,
                'filter_size': (8, 8), 'strides': (4, 4), 'padding': 'VALID',
                'W_init': tf.initializers.GlorotUniform()
            },
            {
                'in_channels': 32, 'n_filter': 64, 'act': tf.nn.relu,
                'filter_size': (4, 4), 'strides': (2, 2), 'padding': 'VALID',
                'W_init': tf.initializers.GlorotUniform()
            },
            {
                'in_channels': 64, 'n_filter': 64, 'act': tf.nn.relu,
                'filter_size': (3, 3), 'strides': (1, 1), 'padding': 'VALID',
                'W_init': tf.initializers.GlorotUniform()
            }
        ]

    ni = tl.layers.Input((1,) + input_shape, name='CNN_Input')
    hi = ni

    for i, kwargs in enumerate(conv_kwargs):
        kwargs['name'] = kwargs.get('name', 'CNN_Layer{}'.format(i + 1))
        hi = tl.layers.Conv2d(**kwargs)(hi)
    no = tl.layers.Flatten(name='Flatten_Layer')(hi)

    return tl.models.Model(inputs=ni, outputs=no)
