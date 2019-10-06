import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.pg.pg import PG
from common.policy_networks import *


''' load environment '''
env = gym.make('CartPole-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
state_shape = env.observation_space.shape
action_shape = (env.action_space.n,)

''' build networks for the algorithm '''
num_hidden_layer = 2 #number of hidden layers for the networks
hidden_dim=32 # dimension of hidden layers for the networks
with tf.name_scope('PG'):
    policy = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim])
net_list = [policy]

''' choose optimizers '''
lr = 0.01  # lr: learning rate of the policy
optimizer = tf.optimizers.Adam(lr)
optimizers_list = [optimizer]

model=PG(net_list, optimizers_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
reward_decay: discounted factor of reward
'''

model.learn(env, train_episodes=3000, test_episodes=1000, max_steps=1000, gamma=0.99,
            seed=2, save_interval=100, mode='train', render=False)
''' 
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
gamma: reward discount factor
seed: random seed
save_interval: timesteps for saving the weights and plotting the results
mode: 'train' or 'test'
render:  if true, visualize the environment
'''


obs = env.reset()
for i in range(100):
    action = model.choose_action(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()