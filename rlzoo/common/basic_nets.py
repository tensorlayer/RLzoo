"""Basic neural networks"""
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense, Input
from gym import spaces
from collections import OrderedDict


def MLP(input_dim, hidden_dim_list, w_init=tf.initializers.Orthogonal(0.2),
        activation=tf.nn.relu, *args, **kwargs):
    """Multiple fully-connected layers for approximation

    :param input_dim: (int) size of input tensor
    :param hidden_dim_list: (list[int]) a list of dimensions of hidden layers
    :param w_init: (callable) initialization method for weights
    :param activation: (callable) activation function of hidden layers

    Return:
        input tensor, output tensor
    """

    l = inputs = Input([None, input_dim])
    for i in range(len(hidden_dim_list)):
        l = Dense(n_units=hidden_dim_list[i], act=activation, W_init=w_init)(l)
    outputs = l

    return inputs, outputs


def MLPModel(input_dim, hidden_dim_list, w_init=tf.initializers.Orthogonal(0.2),
             activation=tf.nn.relu, *args, **kwargs):
    """Multiple fully-connected layers for approximation

    :param input_dim: (int) size of input tensor
    :param hidden_dim_list: (list[int]) a list of dimensions of hidden layers
    :param w_init: (callable) initialization method for weights
    :param activation: (callable) activation function of hidden layers

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

    :param input_shape: (tuple[int]) (H, W, C)
    :param conv_kwargs: (list[param]) list of conv parameters for tl.layers.Conv2d

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
    l = inputs = tl.layers.Input((1,) + input_shape)

    for i, kwargs in enumerate(conv_kwargs):
        # kwargs['name'] = kwargs.get('name', 'cnn_layer{}'.format(i + 1))
        l = tl.layers.Conv2d(**kwargs)(l)
    outputs = tl.layers.Flatten()(l)

    return inputs, outputs


def CNNModel(input_shape, conv_kwargs=None):
    """Multiple convolutional layers for approximation
    Default setting is equal to architecture used in DQN

    :param input_shape: (tuple[int]) (H, W, C)
    :param conv_kwargs: (list[param]) list of conv parameters for tl.layers.Conv2d

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


def CreateInputLayer(state_space, conv_kwargs=None):
    def CreateSingleInput(single_state_space):
        single_state_shape = single_state_space.shape
        # build structure
        if len(single_state_shape) == 1:
            l = inputs = Input((None,) + single_state_shape, name='input_layer')
        else:
            with tf.name_scope('CNN'):
                inputs, l = CNN(single_state_shape, conv_kwargs=conv_kwargs)
        return inputs, l, single_state_shape

    if isinstance(state_space, spaces.Dict):
        input_dict, layer_dict, shape_dict = OrderedDict(), OrderedDict(), OrderedDict()
        for k, v in state_space.spaces.items():
            input_dict[k], layer_dict[k], shape_dict[k] = CreateSingleInput(v)
        return input_dict, layer_dict, shape_dict
    if isinstance(state_space, spaces.Space):
        return CreateSingleInput(state_space)
    else:
        raise ValueError('state space error')
