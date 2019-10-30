import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.a3c.a3c import A3C
from common.value_networks import *
from common.policy_networks import *


''' load environment '''
env_id='BipedalWalker-v2'
env = gym.make(env_id).unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
action_shape = env.action_space.shape
state_shape = env.observation_space.shape

''' build networks for the algorithm '''
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim=64 # dimension of hidden layers for the networks
num_workers = 2
net_list2 = []
for i in range(num_workers+1):
    with tf.name_scope('A3C'):
        with tf.name_scope('Actor'):
            actor = StochasticPolicyNetwork(env.observation_space, env.action_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
        with tf.name_scope('Critic'):
            critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer*[hidden_dim])
    net_list = [actor, critic]
    net_list2.append(net_list)

''' choose optimizers '''
actor_lr, critic_lr = 5e-5, 1e-4 # learning rate
a_optimizer = tf.optimizers.RMSprop(actor_lr)
c_optimizer = tf.optimizers.RMSprop(critic_lr)
optimizers_list= [a_optimizer, c_optimizer]

model=A3C(net_list2, optimizers_list)
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
replay_buffer_capacity: the size of buffer for storing explored samples
action_range: value of each action in [-action_range, action_range]
soft_q_lr: learning rate of the Q network
policy_lr: learning rate of the policy network
alpha_lr: learning rate of the variable alpha
'''


env_list=[]
for i in range(num_workers):
    env_list.append(gym.make(env_id).unwrapped)
model.learn(env_list, train_episodes=100, test_episodes=1000, max_steps=150, number_workers=num_workers, update_itr=10,
        gamma=0.99, entropy_beta=0.005 , actor_lr=5e-5, critic_lr=1e-4, seed=2, save_interval=500, mode='train')
''' 
full list of parameters for training
---------------------------------------
env_list: a list of same learning environments
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
number_workers: manually set number of workers
update_itr: update global policy after several episodes
gamma: reward discount factor
entropy_beta: factor for entropy boosted exploration
actor_lr: learning rate for actor
critic_lr: learning rate for critic
seed: random seed
save_interval: timesteps for saving the weights and plotting the results
mode: train or test
'''

