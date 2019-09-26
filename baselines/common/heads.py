"""Head of networks. Receive latent variables outputted by backbone and return
values or the parameters of the policy distribution"""
import tensorflow as tf
import tensorlayer as tl


def mlp(input_shape, num_output, num_layers=1, num_hidden=64,
        activation=tf.nn.relu, activation_output=None, scale=1e-2):
    """Multiple fully-connected layers for approximation

    Args:
        input_shape (int): shape of input tensor
        num_output (int): size of the last fully-connected layer
        num_layers (int): number of fully-connected layers
        num_hidden (int): size of fully-connected layers
        activation (callable): activation function of hidden
        activation_output (callable): activation function of output
        scale (float): scale for orthogonal initialization
    Return:
        tl.model.Model
    """
    ni = tl.layers.Input((1, input_shape), name='observation')
    hi = ni

    for i in range(num_layers):
        name = 'HeadLayer{}'.format(i + 1)
        hi = tl.layers.Dense(n_units=num_hidden, act=activation,
                             W_init=tf.initializers.Orthogonal(scale),
                             in_channels=input_shape, name=name)(hi)
        input_shape = num_hidden

    output = tl.layers.Dense(n_units=num_output, act=activation_output,
                             W_init=tf.initializers.Orthogonal(scale),
                             in_channels=input_shape, name='HeadOutput')(hi)
    return tl.models.Model(inputs=ni, outputs=output, name='MLPHead')
