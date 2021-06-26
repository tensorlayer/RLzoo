from rlzoo.algorithms.pg.pg import PG
from rlzoo.common.policy_networks import *
import gym

""" load environment """
env = gym.make('CartPole-v0').unwrapped
# env = gym.make('Pendulum-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
obs_space = env.observation_space
act_space = env.action_space

# reproducible
seed = 2
np.random.seed(seed)
tf.random.set_seed(seed)
env.seed(seed)

""" build networks for the algorithm """
name = 'pg'
num_hidden_layer = 1  # number of hidden layers for the networks
hidden_dim = 32  # dimension of hidden layers for the networks

policy_net = StochasticPolicyNetwork(obs_space, act_space, num_hidden_layer * [hidden_dim])
net_list = [policy_net]

""" choose optimizers """
learning_rate = 0.02
policy_optimizer = tf.optimizers.Adam(learning_rate)
optimizers_list = [policy_optimizer]

model = PG(net_list, optimizers_list)
""" 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
"""

model.learn(env, train_episodes=200, max_steps=200, save_interval=20, mode='train', render=False, gamma=0.95)
"""
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: time steps for saving
mode: train or test
render: render each step
gamma: reward decay
"""

# test
model.learn(env, test_episodes=100, max_steps=200, mode='test', render=True)
