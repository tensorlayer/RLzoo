"""Backbone networks for feature extraction"""
import math

import tensorflow as tf
import tensorlayer as tl


def mlp(input_shape, num_layers=2, num_hidden=64, activation=tf.tanh):
    """Multiple fully-connected layers for approximation

    Args:
        input_shape (int): shape of input tensor
        num_layers (int): number of fully-connected layers
        num_hidden (int): size of fully-connected layers
        activation (callable): activation function
    Return:
        tl.model.Model
    """
    ni = tl.layers.Input((1, input_shape), name='observation')
    hi = ni

    for i in range(num_layers):
        name = 'BackboneLayer{}'.format(i + 1)
        hi = tl.layers.Dense(n_units=num_hidden, act=activation,
                             W_init=tf.initializers.Orthogonal(math.sqrt(2)),
                             in_channels=input_shape, name=name)(hi)
        input_shape = num_hidden

    return tl.models.Model(inputs=ni, outputs=hi, name='MLPBackbone')


def cnn(input_shape, conv_kwargs=None):
    """Multiple fully-connected layers for approximation
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

    ni = tl.layers.Input((1, ) + input_shape, name='observation')
    hi = ni

    for i, kwargs in enumerate(conv_kwargs):
        kwargs['name'] = kwargs.get('name', 'BackboneLayer{}'.format(i + 1))
        hi = tl.layers.Conv2d(**kwargs)(hi)

    return tl.models.Model(inputs=ni, outputs=hi, name='CNNBackbone')
