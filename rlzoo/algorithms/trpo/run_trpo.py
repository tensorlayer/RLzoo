from rlzoo.common.utils import set_seed
from rlzoo.algorithms.trpo.trpo import TRPO
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
import gym

""" load environment """
env = gym.make('Pendulum-v0').unwrapped

# reproducible
seed = 2
set_seed(seed, env)

""" build networks for the algorithm """
name = 'TRPO'
hidden_dim = 64
num_hidden_layer = 2
critic = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer, name=name + '_value')

actor = StochasticPolicyNetwork(env.observation_space, env.action_space, [hidden_dim] * num_hidden_layer,
                                output_activation=tf.nn.tanh, name=name + '_policy')
net_list = critic, actor

critic_lr = 1e-3
optimizers_list = [tf.optimizers.Adam(critic_lr)]

""" create model """
model = TRPO(net_list, optimizers_list, damping_coeff=0.1, cg_iters=10, delta=0.01)
"""
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
damping_coeff: Artifact for numerical stability
cg_iters: Number of iterations of conjugate gradient to perform
delta: KL-divergence limit for TRPO update.
"""

model.learn(env, mode='train', render=False, train_episodes=2000, max_steps=200, save_interval=100,
            gamma=0.9, batch_size=256, backtrack_iters=10, backtrack_coeff=0.8, train_critic_iters=80)
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
batch_size: update batch size
backtrack_iters: Maximum number of steps allowed in the backtracking line search
backtrack_coeff: How far back to step during backtracking line search
train_critic_iters: critic update iteration steps
"""

model.learn(env, test_episodes=100, max_steps=200, mode='test', render=True)
