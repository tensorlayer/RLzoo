import gym

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
name='ac'
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim=64 # dimension of hidden layers for the networks
critic = MlpValueNetwork(state_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_critic')
actor = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_actor')
net_list = [actor, critic]

model=AC(net_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
a_lr: learning rate of the actor
c_lr: learning rate of the critic
gamma: discounted factor of reward
'''

model.learn(env, train_episodes=100, test_episodes=1000, max_steps=1000,
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