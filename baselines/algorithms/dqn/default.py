import tensorflow as tf
import tensorlayer as tl

from common import math_utils


def atari(env, **kwargs):
    in_dim = env.observation_space.shape
    act_dim = env.action_space.n
    params = dict(
        batch_size=32,
        double_q=True,
        buffer_size=10000,
        exploration_rate=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_alpha=0.6,
        prioritized_beta0=0.4,
        dueling=True
    )
    params.update(kwargs)

    if params.get('network') is None:
        params['network'] = CNNQNet(in_dim, act_dim, params.pop('dueling'))
    if params.get('optimizer') is None:
        params['optimizer'] = tf.optimizers.Adam(1e-4,
                                                 epsilon=1e-5, clipnorm=10)

    return {}, params


def classic_control(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    params = dict(
        batch_size=32,
        double_q=True,
        buffer_size=1000,
        exploration_rate=0.2,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=200,
        target_network_update_freq=50,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_alpha=0.6,
        prioritized_beta0=0.4,
        dueling=True
    )
    params.update(kwargs)
    if params.get('network') is None:
        params['network'] = MLPQNet(in_dim, act_dim, params.pop('dueling'))
    if params.get('optimizer') is None:
        params['optimizer'] = tf.optimizers.Adam(5e-3, epsilon=1e-5)
    return {}, params


class CNNQNet(tl.models.Model):
    def __init__(self, in_dim, act_dim, dueling):
        super().__init__()
        self.dueling = dueling
        with tf.name_scope('DQN'):
            with tf.name_scope('CNN'):
                self.cnn = CNN(in_dim)
            latent_shape = self.cnn.outputs[0].shape
            mlp_in_shape = math_utils.flatten_dims(latent_shape)
            with tf.name_scope('QValue'):
                self.qmlp = MLP(mlp_in_shape, act_dim, 1, 256, scale=1)
            if dueling:
                with tf.name_scope('Value'):
                    self.vmlp = MLP(mlp_in_shape, 1, 1, 256, scale=1)

    def forward(self, obv):
        obv = tf.cast(obv, tf.float32) / 255.0
        mlp_in = tl.layers.flatten_reshape(self.cnn(obv))
        q_out = self.qmlp(mlp_in)
        if self.dueling:
            v_out = self.vmlp(mlp_in)
            q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
        return q_out


class MLPQNet(tl.models.Model):
    def __init__(self, in_dim, act_dim, dueling):
        super().__init__()
        self.dueling = dueling
        with tf.name_scope('DQN'):
            with tf.name_scope('MLP'):
                self.mlp = MLP(in_dim, 64, 1, 64, tf.nn.tanh, tf.nn.tanh, 1)
            latent_shape = math_utils.flatten_dims(self.mlp.outputs[0].shape)
            with tf.name_scope('QValue'):
                self.qmlp = MLP(latent_shape, act_dim, 0, scale=1)
            if dueling:
                with tf.name_scope('Value'):
                    self.vmlp = MLP(latent_shape, 1, 0, scale=1)

    def forward(self, obv):
        obv = tf.cast(obv, tf.float32)
        latent = self.mlp(obv)
        q_out = self.qmlp(latent)
        if self.dueling:
            v_out = self.vmlp(latent)
            q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
        return q_out


# ========= put here temporally =========
def CNN(input_shape, conv_kwargs=None):
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

    ni = tl.layers.Input((1, ) + input_shape, name='CNN_Input')
    hi = ni

    for i, kwargs in enumerate(conv_kwargs):
        hi = tl.layers.Conv2d(**kwargs)(hi)

    return tl.models.Model(inputs=ni, outputs=hi)


def MLP(input_dim, output_dim, num_layers=1, num_hidden=64,
        activation=tf.nn.relu, activation_output=None, scale=1e-2):
    """Multiple fully-connected layers for approximation

    Args:
        input_dim (int): size of input tensor
        output_dim (int): size of the last fully-connected layer
        num_layers (int): number of fully-connected layers
        num_hidden (int): size of fully-connected layers
        activation (callable): activation function of hidden
        activation_output (callable): activation function of output
        scale (float): scale for orthogonal initialization
    Return:
        tl.model.Model
    """
    ni = tl.layers.Input((1, input_dim))
    hi = ni

    for i in range(num_layers):
        hi = tl.layers.Dense(n_units=num_hidden, act=activation,
                             W_init=tf.initializers.Orthogonal(scale),
                             in_channels=input_dim)(hi)
        input_dim = num_hidden

    output = tl.layers.Dense(n_units=output_dim, act=activation_output,
                             W_init=tf.initializers.Orthogonal(scale),
                             in_channels=input_dim)(hi)
    return tl.models.Model(inputs=ni, outputs=output)
# ========= put here temporally =========
