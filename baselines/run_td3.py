import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.td3.td3 import TD3
from common.value_networks import *
from common.policy_networks import *



env = gym.make('Pendulum-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

model=TD3(QNetwork=MlpQNetwork, PolicyNetwork=DeterministicPolicyNetwork, state_dim=state_dim, action_dim=action_dim)
''' 
full list of arguments for the algorithm
----------------------------------------
QNetwork: choose network structure for the Q-value network
PolicyNetwork: choose MLP, CNN or RNN and deterministic or stochastic for the policy network
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
replay_buffer_capacity: the size of buffer for storing explored samples
policy_target_update_interval: delayed interval for updating the target policy
action_range: value of each action in [-action_range, action_range]
hidden_dim: dimension of hidden layers for the networks
num_hidden_layer: number of hidden layers for the networks
q_lr: learning rate of the Q network
policy_lr: learning rate of the policy network
'''

model.learn(env, train_episodes=100)
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