import gym
import tensorflow as tf
# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.ac.ac import AC
from common.value_networks import *
from common.policy_networks import *


''' load environment '''
env = gym.make('CartPole-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
state_shape = env.observation_space.shape
action_shape = (env.action_space.n,)

''' build networks for the algorithm '''
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim = 64 # dimension of hidden layers for the networks
with tf.name_scope('AC'):
        with tf.name_scope('Critic'):
                critic = ValueNetwork(state_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
        with tf.name_scope('Actor'):
                actor = DeterministicPolicyNetwork(env.observation_space, env.action_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
net_list = [actor, critic]

''' choose optimizers '''
a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
a_optimizer = tf.optimizers.Adam(a_lr)
c_optimizer = tf.optimizers.Adam(c_lr)
optimizers_list=[a_optimizer, c_optimizer]

model=AC(net_list, optimizers_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
gamma: discounted factor of reward
'''

model.learn(env, train_episodes=1000, test_episodes=1000, max_steps=1000,
        seed=2, save_interval=100, mode='train', render=False)
''' 
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
seed: random seed
save_interval: timesteps for saving the weights and plotting the results
mode: 'train' or 'test'
render:  if true, visualize the environment
'''


obs = env.reset()
for i in range(100):
    action = model.get_action(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()