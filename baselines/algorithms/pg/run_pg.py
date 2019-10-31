import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.pg.pg import PG

from common.policy_networks import *

''' load environment '''
env = gym.make('CartPole-v0').unwrapped
# env = gym.make('Pendulum-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
obs_space = env.observation_space
act_space = env.action_space

''' build networks for the algorithm '''
name = 'pg'
num_hidden_layer = 1  # number of hidden layers for the networks
hidden_dim = 64  # dimension of hidden layers for the networks

policy_net = StochasticPolicyNetwork(obs_space, act_space, num_hidden_layer * [hidden_dim],
                                     output_activation=tf.nn.tanh)
net_list = [policy_net]

''' choose optimizers '''
learning_rate = 0.02
policy_optimizer = tf.optimizers.Adam(learning_rate)
optimizers_list = [policy_optimizer]

model = PG(net_list, optimizers_list)
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
'''

model.learn(env, train_episodes=300, max_steps=3000, save_interval=100,
            mode='train', render=False, gamma=0.95, seed=2,)
"""
full list of parameters for training
---------------------------------------
learn function
env: learning environment
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: timesteps for saving
mode: train or test
render: render each step
gamma: reward decay
seed: random seed
"""

# test
model.learn(env, test_episodes=200, max_steps=3000,
            mode='test', render=True,)