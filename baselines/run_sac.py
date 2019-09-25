import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.sac.sac import SAC
from common.value_networks import *
from common.policy_networks import *



env = gym.make('Pendulum-v0').unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

model=SAC(QNetwork=MlpQNetwork, PolicyNetwork=StochasticPolicyNetwork, state_dim=state_dim, action_dim=action_dim)
''' 
full list of arguments for the algorithm
----------------------------------------
QNetwork: choose MLP, CNN or RNN for the Q-value network
PolicyNetwork: choose MLP, CNN or RNN and deterministic or stochastic for the policy network
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
replay_buffer_capacity: the size of buffer for storing explored samples
action_range=1., hidden_dim=32, num_hidden_layer=3, soft_q_lr=3e-4, policy_lr=3e-4, alpha_lr=3e-4
action_range: value of each action in [-action_range, action_range]
hidden_dim: dimension of hidden layers for the networks
num_hidden_layer: number of hidden layers for the networks
soft_q_lr: learning rate of the Q network
policy_lr: learning rate of the policy network
alpha_lr: learning rate of the variable alpha
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
policy_target_update_interval: delayed update for the policy network and target networks
reward_scale: value range of reward
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