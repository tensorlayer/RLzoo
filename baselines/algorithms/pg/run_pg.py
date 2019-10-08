import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.pg.pg import PolicyGradient
from common.policy_networks import *

from common.policy_networks import *

''' load environment '''
env = gym.make('CartPole-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
action_shape = env.action_space.n,
state_shape = env.observation_space.shape

''' build networks for the algorithm '''
name = 'pg'
num_hidden_layer = 1  # number of hidden layers for the networks
hidden_dim = 64  # dimension of hidden layers for the networks

policy_net = DeterministicPolicyNetwork(state_shape, action_shape, num_hidden_layer * [hidden_dim],
                                        name=name + '_policy')
net_list = [policy_net]

''' choose optimizers '''
learning_rate = 0.02
policy_optimizer = tf.optimizers.Adam(learning_rate)
optimizers_list = [policy_optimizer]

model = PolicyGradient(net_list, optimizers_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
'''

model.learn(env, train_episodes=300, test_episodes=200, max_steps=3000, save_interval=100,
            mode='train', render=False, gamma=0.95, seed=2)
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
observation = env.reset()
for step in range(3000):
    env.render()
    action = model.choose_action_greedy(observation)
    observation, reward, done, info = env.step(action)
    if done: break
env.close()
