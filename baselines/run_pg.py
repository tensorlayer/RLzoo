import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.pg.pg import PolicyGradient
from common.policy_networks import *


''' load environment '''
env = gym.make('CartPole-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
state_shape = env.observation_space.shape
action_shape = (env.action_space.n,)

''' build networks for the algorithm '''
name='ac'
num_hidden_layer = 2 #number of hidden layers for the networks
hidden_dim=32 # dimension of hidden layers for the networks
policy = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_policy')
net_list = [policy]

model=PolicyGradient(net_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
learning_rate: learning rate of the actor
reward_decay: discounted factor of reward
'''

model.learn(env, train_episodes=3000, test_episodes=1000, max_steps=1000, lr=0.02, gamma=0.99,
            seed=2, save_interval=100, mode='train', render=False)
''' 
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
lr: policy learning rate
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