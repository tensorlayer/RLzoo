import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.td3.td3 import TD3
from common.value_networks import *
from common.policy_networks import *

''' load environment '''
env = gym.make('Pendulum-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
action_shape = env.action_space.shape
state_shape = env.observation_space.shape

''' build networks for the algorithm '''
name='td3'
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim=64 # dimension of hidden layers for the networks
q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_q1')
q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_q2')
target_q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_target_q1')
target_q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_target_q2')
policy_net = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_policy')
target_policy_net = DeterministicPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_target_policy')
net_list = [q_net1, q_net2, target_q_net1, target_q_net2, policy_net, target_policy_net]

model=TD3(net_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
replay_buffer_capacity: the size of buffer for storing explored samples
policy_target_update_interval: delayed interval for updating the target policy
action_range: value of each action in [-action_range, action_range]
q_lr: learning rate of the Q network
policy_lr: learning rate of the policy network
'''

model.learn(env, train_episodes=100, max_steps=150, batch_size=64, explore_steps=500, update_itr=3, 
reward_scale = 1. , seed=2, save_interval=10, explore_noise_scale = 1.0, eval_noise_scale = 0.5, mode='train')
''' 
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
batch_size:  udpate batchsize
explore_steps:  for random action sampling in the beginning of training
update_itr: repeated updates for single step
reward_scale: value range of reward
seed: random seed
save_interval: timesteps for saving the weights and plotting the results
explore_noise_scale: range of action noise for exploration
eval_noise_scale: range of action noise for evaluation of action value
mode: 'train' or 'test'
'''


obs = env.reset()
for i in range(1000):
    action = model.get_action(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()