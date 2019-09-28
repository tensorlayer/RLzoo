import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.sac.sac import SAC
from common.value_networks import *
from common.policy_networks import *


''' load environment '''
env = gym.make('Pendulum-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
action_shape = env.action_space.shape
state_shape = env.observation_space.shape

''' build networks for the algorithm '''
name='sac'
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim=64 # dimension of hidden layers for the networks
soft_q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_q1')
soft_q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_q2')
target_soft_q_net1 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_target_q1')
target_soft_q_net2 = MlpQNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_target_q2')
policy_net = StochasticPolicyNetwork(state_shape, action_shape, hidden_dim_list=num_hidden_layer*[hidden_dim], name=name+'_policy')
net_list = [soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2, policy_net]

model=SAC(net_list, state_dim=state_shape[0], action_dim=action_shape[0])
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
replay_buffer_capacity: the size of buffer for storing explored samples
action_range: value of each action in [-action_range, action_range]
soft_q_lr: learning rate of the Q network
policy_lr: learning rate of the policy network
alpha_lr: learning rate of the variable alpha
'''

model.learn(env, train_episodes=100, max_steps=150, batch_size=64, explore_steps=500, \
update_itr=3, policy_target_update_interval = 3,  reward_scale = 1. , seed=2, save_interval=20, \
mode='train', AUTO_ENTROPY = True, DETERMINISTIC = False)
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
policy_target_update_interval: delayed update for the policy network and target networks
reward_scale: value range of reward
seed: random seed
save_interval: timesteps for saving the weights and plotting the results
mode: 'train'  or 'test'
AUTO_ENTROPY: automatically udpating variable alpha for entropy
DETERMINISTIC: stochastic action policy if False, otherwise deterministic
'''


obs = env.reset()
for i in range(1000):
    action = model.get_action(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()