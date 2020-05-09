from rlzoo.common.utils import make_env, set_seed
from rlzoo.algorithms.ppo_clip.ppo_clip import PPO_CLIP
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
import gym


""" load environment """
env = gym.make('Pendulum-v0').unwrapped

# reproducible
seed = 1
set_seed(seed, env)

""" build networks for the algorithm """
name = 'PPO_CLIP'
hidden_dim = 64
num_hidden_layer = 2
critic = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer, name=name + '_value')

actor = StochasticPolicyNetwork(env.observation_space, env.action_space, [hidden_dim] * num_hidden_layer,
                                output_activation=tf.nn.tanh, name=name + '_policy')
net_list = critic, actor

""" create model """
actor_lr = 1e-4
critic_lr = 2e-4
optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]

model = PPO_CLIP(net_list, optimizers_list,)
"""
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
epsilon: clip parameter
"""

model.learn(env, train_episodes=500, max_steps=200, save_interval=50, gamma=0.9,
            mode='train', render=False, batch_size=32, a_update_steps=10, c_update_steps=10)

"""
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: time steps for saving
gamma: reward discount factor
mode: train or test
render: render each step
batch_size: UPDATE batch size
a_update_steps: actor update iteration steps
c_update_steps: critic update iteration steps
:return: None
"""
model.learn(env, test_episodes=100, max_steps=200, mode='test', render=True)

