import gym

from rlzoo.algorithms.dqn.dqn import DQN
from rlzoo.algorithms.dqn.default import *
from rlzoo.common.value_networks import *
import gym

""" load environment """
env = gym.make('CartPole-v0').unwrapped

obs_space = env.observation_space
act_space = env.action_space

# reproducible
seed = 2
set_seed(seed, env)

in_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
""" build networks for the algorithm """
name = 'DQN'
Q_net = QNetwork(env.observation_space, env.action_space, [64], activation=tf.nn.tanh,
                 state_only=True, dueling=True)
net_list = [Q_net]

""" create model """
optimizer = tf.optimizers.Adam(5e-3, epsilon=1e-5)
optimizers_list = [optimizer]
model = DQN(net_list, optimizers_list,
            double_q=True,
            dueling=True,
            buffer_size=10000,
            prioritized_replay=False,
            prioritized_alpha=0.6,
            prioritized_beta0=0.4)
""" 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
replay_buffer_size: the size of buffer for storing explored samples
tau: soft update factor
"""

model.learn(env, mode='train', render=False,
            train_episodes=1000,
            test_episodes=10,
            max_steps=200,
            save_interval=1e3,
            batch_size=32,
            exploration_rate=0.2,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=200,
            target_network_update_freq=50,
            gamma=0.99, )
"""
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: time steps for saving
explore_steps: for random action sampling in the beginning of training
mode: train or test mode
render: render each step
batch_size: update batch size
gamma: reward decay factor
noise_scale: range of action noise for exploration
noise_scale_decay: noise scale decay factor
"""

model.learn(env, mode='test', render=True,
            test_episodes=10,
            batch_size=32,
            exploration_rate=0.2,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=200,
            target_network_update_freq=50,
            gamma=0.99, )
