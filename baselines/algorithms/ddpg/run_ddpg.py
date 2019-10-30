import gym

# from common.env_wrappers import DummyVecEnv
from algorithms.ddpg.ddpg import DDPG
from common.value_networks import *
from common.policy_networks import *

''' load environment '''
env = gym.make('Pendulum-v0').unwrapped

obs_space = env.observation_space
act_space = env.action_space

''' build networks for the algorithm '''
name = 'ddpg'
num_hidden_layer = 1  # number of hidden layers for the networks
hidden_dim = 30  # dimension of hidden layers for the networks

actor = DeterministicPolicyNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer)
critic = QNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer)

actor_target = DeterministicPolicyNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer, trainable=False)

critic_target = QNetwork(obs_space, act_space, [hidden_dim] * num_hidden_layer, trainable=False)

net_list = [critic, critic_target, actor, actor_target]

''' create model '''
actor_lr = 1e-3
critic_lr = 2e-3
optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
replay_buffer_size = 10000
model = DDPG(net_list, optimizers_list, replay_buffer_size)
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
state_dim: dimension of state for the environment
action_dim: dimension of action for the environment
a_bounds: a list of [min_action, max_action] action bounds for the environment
replay_buffer_size: the size of buffer for storing explored samples
tau: soft update factor
var: control exploration
'''

model.learn(env, train_episodes=200, test_episodes=100, max_steps=200, save_interval=10,
              mode='train', render=False, batch_size=32, gamma=0.9, seed=1)
'''
full list of parameters for training
---------------------------------------
learn function
env: learning environment
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: time steps for saving
mode: train or test mode
render: render each step
batch_size: update batch size
gamma: reward decay factor
seed: random seed
reward_shaping: reward shaping function
:return: None
'''

obs = env.reset()
s = env.reset()
for i in range(200):
    env.render()
    s, r, done, info = env.step(model.choose_action(s))
    if done:
        break
env.close()
