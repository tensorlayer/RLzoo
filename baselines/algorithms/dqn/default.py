import tensorflow as tf
import tensorlayer as tl

from baselines.common import math_utils


def atari(env, **kwargs):
    in_dim = env.observation_space.shape
    act_dim = env.action_space.n
    params = dict(
        grad_norm=10,
        batch_size=32,
        double_q=True,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        dueling=True,
        atom_num=1,
        min_value=-10,
        max_value=10,
        ob_scale=1 / 255.0
    )
    params.update(kwargs)

    network = CNNQNet(in_dim, act_dim,
                      params['atom_num'], params.pop('dueling'))
    optimizer = tf.optimizers.Adam(1e-4, epsilon=1e-5)
    params.update(network=network, optimizer=optimizer)
    return params


def classic_control(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    params = dict(
        grad_norm=10,
        batch_size=100,
        double_q=True,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=200,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        dueling=True,
        atom_num=1,
        min_value=-10,
        max_value=10,
        ob_scale=1
    )
    params.update(kwargs)
    network = MLPQNet(in_dim, act_dim,
                      params['atom_num'], params.pop('dueling'))
    optimizer = tf.optimizers.Adam(1e-3, epsilon=1e-5)
    params.update(network=network, optimizer=optimizer)
    return params


class CNNQNet(tl.models.Model):
    def __init__(self, in_dim, act_dim, atom_num, dueling):
        super().__init__()
        self.atom_num = atom_num
        self.dueling = dueling
        with tf.name_scope('DQN'):
            with tf.name_scope('CNN'):
                self.cnn = CNN(in_dim)
            latent_shape = self.cnn.outputs[0].shape
            mlp_in_shape = math_utils.flatten_dims(latent_shape)
            q_out_shape = act_dim * atom_num
            with tf.name_scope('QValue'):
                self.qmlp = MLP(mlp_in_shape, q_out_shape, 1, 256, scale=1)
            if dueling:
                with tf.name_scope('Value'):
                    self.vmlp = MLP(mlp_in_shape, atom_num, 1, 256, scale=1)

    def forward(self, obv):
        batch_size = obv.shape[0]
        mlp_in = tl.layers.flatten_reshape(self.cnn(obv))
        q_out = self.qmlp(mlp_in)
        if self.atom_num == 1:
            if self.dueling:
                v_out = self.vmlp(mlp_in)
                q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
            return q_out
        else:
            q_shape = (batch_size, -1, self.atom_num)
            q_out = tf.reshape(q_out, q_shape)
            if self.dueling:
                v_out = tf.reshape(self.vmlp(mlp_in), q_shape)
                q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
            logprobs = tf.nn.log_softmax(q_out, -1)
            return logprobs


class MLPQNet(tl.models.Model):
    def __init__(self, in_dim, act_dim, atom_num, dueling):
        super().__init__()
        self.atom_num = atom_num
        self.dueling = dueling
        with tf.name_scope('DQN'):
            with tf.name_scope('MLP'):
                self.mlp = MLP(in_dim, 64, 1, 64, tf.nn.tanh, tf.nn.tanh, 1)
            latent_shape = self.mlp.outputs[0].shape
            q_out_shape = act_dim * atom_num
            with tf.name_scope('QValue'):
                self.qmlp = MLP(latent_shape, q_out_shape, 0, scale=1)
            if dueling:
                with tf.name_scope('Value'):
                    self.vmlp = MLP(latent_shape, atom_num, 0, scale=1)

    def forward(self, obv):
        batch_size = obv.shape[0]
        latent = self.mlp(obv)
        q_out = self.qmlp(latent)
        if self.atom_num == 1:
            if self.dueling:
                v_out = self.vmlp(latent)
                q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
            return q_out
        else:
            q_shape = (batch_size, -1, self.atom_num)
            q_out = tf.reshape(q_out, q_shape)
            if self.dueling:
                v_out = tf.reshape(self.vmlp(latent), q_shape)
                q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
            logprobs = tf.nn.log_softmax(q_out, -1)
            return logprobs


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
