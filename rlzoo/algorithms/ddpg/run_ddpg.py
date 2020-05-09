from rlzoo.common.utils import make_env, set_seed
from rlzoo.algorithms.ddpg.ddpg import DDPG
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
import gym

""" load environment """
env = gym.make('Pendulum-v0').unwrapped

obs_space = env.observation_space
act_space = env.action_space

# reproducible
seed = 2
set_seed(seed, env)

""" build networks for the algorithm """
name = 'DDPG'
num_hidden_layer = 2  # number of hidden layers for the networks
hidden_dim = 64  # dimension of hidden layers for the networks

actor = DeterministicPolicyNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer)
critic = QNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer)

actor_target = DeterministicPolicyNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer, trainable=False)

critic_target = QNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer, trainable=False)

net_list = [critic, critic_target, actor, actor_target]

""" create model """
actor_lr = 1e-3
critic_lr = 2e-3
optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
replay_buffer_size = 10000
model = DDPG(net_list, optimizers_list, replay_buffer_size)
""" 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
replay_buffer_size: the size of buffer for storing explored samples
tau: soft update factor
"""

model.learn(env, train_episodes=100, max_steps=200, save_interval=10,
            mode='train', render=False, batch_size=32, gamma=0.9, noise_scale=1., noise_scale_decay=0.995)
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

model.learn(env, test_episodes=10, max_steps=200, mode='test', render=True)

