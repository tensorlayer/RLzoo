import tensorflow as tf
import tensorlayer as tl

from rlzoo.common import basic_nets
from rlzoo.common.utils import set_seed

''' 
full list of algorithm parameters (alg_params)
-----------------------------------------------
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
double_q (bool): if True double DQN will be used
dueling (bool): if True dueling value estimation will be used
exploration_rate (float): fraction of entire training period over
    which the exploration rate is annealed
exploration_final_eps (float): final value of random action probability
batch_size (int): size of a batched sampled from replay buffer for training
train_freq (int): update the model every `train_freq` steps
learning_starts (int): how many steps of the model to collect transitions
                        for before learning starts
target_network_update_freq (int): update the target network every
                                    `target_network_update_freq` steps
buffer_size (int): size of the replay buffer
prioritized_replay (bool): if True prioritized replay buffer will be used.
prioritized_alpha (float): alpha parameter for prioritized replay
prioritized_beta0 (float): beta parameter for prioritized replay
mode (str): train or test
-----------------------------------------------
'''

def atari(env, default_seed=False, **kwargs):
    if default_seed:
        seed = 2
        set_seed(seed, env) # reproducible

    in_dim = env.observation_space.shape
    act_dim = env.action_space.n
    params = dict(
        number_timesteps=int(1e7),   # for raw-pixel
        test_episodes=10, 
        save_path=None,
        save_interval=1e4,
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
    return dict(), params


def classic_control(env, default_seed=False, **kwargs):
    if default_seed:
        seed = 2
        set_seed(seed, env) # reproducible

    in_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    params = dict(
        number_timesteps=int(1e4), 
        test_episodes=10, 
        save_path=None,
        save_interval=1e3,
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
        dueling=True,
    )
    params.update(kwargs)
    if params.get('network') is None:
        params['network'] = MLPQNet(in_dim, act_dim, params.pop('dueling'))
    if params.get('optimizer') is None:
        params['optimizer'] = tf.optimizers.Adam(5e-3, epsilon=1e-5)
    return  dict(), params


class CNNQNet(tl.models.Model):
    def __init__(self, in_dim, act_dim, dueling):
        super().__init__()
        self.dueling = dueling
        with tf.name_scope('DQN'):
            with tf.name_scope('CNN'):
                self.cnn = basic_nets.CNNModel(in_dim)
            mlp_in_shape = self.cnn.outputs[0].shape[0]
            with tf.name_scope('QValue'):
                hidden_dim = 256
                self.preq = tl.layers.Dense(
                    hidden_dim, tf.nn.relu,
                    tf.initializers.Orthogonal(1.0),
                    in_channels=mlp_in_shape
                )
                self.qout = tl.layers.Dense(
                    act_dim, None,
                    tf.initializers.Orthogonal(1.0),
                    in_channels=hidden_dim
                )
            if dueling:
                with tf.name_scope('Value'):
                    hidden_dim = 256
                    self.prev = tl.layers.Dense(
                        hidden_dim, tf.nn.relu,
                        tf.initializers.Orthogonal(1.0),
                        in_channels=mlp_in_shape
                    )
                    self.vout = tl.layers.Dense(
                        1, None,
                        tf.initializers.Orthogonal(1.0),
                        in_channels=hidden_dim
                    )

    def forward(self, obv):
        obv = tf.cast(obv, tf.float32) / 255.0
        mlp_in = tl.layers.flatten_reshape(self.cnn(obv))
        q_out = self.qout(self.preq(mlp_in))
        if self.dueling:
            v_out = self.vout(self.prev(mlp_in))
            q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
        return q_out


class MLPQNet(tl.models.Model):
    def __init__(self, in_dim, act_dim, dueling):
        super().__init__()
        self.dueling = dueling
        hidden_dim = 64
        with tf.name_scope('DQN'):
            with tf.name_scope('MLP'):
                self.mlp = tl.layers.Dense(
                    hidden_dim, tf.nn.tanh,
                    tf.initializers.Orthogonal(1.0),
                    in_channels=in_dim
                )
            with tf.name_scope('QValue'):
                self.qmlp = tl.layers.Dense(
                    act_dim, None,
                    tf.initializers.Orthogonal(1.0),
                    in_channels=hidden_dim
                )
            if dueling:
                with tf.name_scope('Value'):
                    self.vmlp = tl.layers.Dense(
                        1, None,
                        tf.initializers.Orthogonal(1.0),
                        in_channels=hidden_dim
                    )

    def forward(self, obv):
        obv = tf.cast(obv, tf.float32)
        latent = self.mlp(obv)
        q_out = self.qmlp(latent)
        if self.dueling:
            v_out = self.vmlp(latent)
            q_out = v_out + q_out - tf.reduce_mean(q_out, 1, True)
        return q_out
