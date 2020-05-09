from rlzoo.algorithms.a3c.a3c import A3C
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
import gym

""" load environment """
env_id = 'BipedalWalker-v2'
env = gym.make(env_id).unwrapped
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized/wrapped environment to run
action_shape = env.action_space.shape
state_shape = env.observation_space.shape
# reproducible
seed = 2
np.random.seed(seed)
tf.random.set_seed(seed)
env.seed(seed)

""" build networks for the algorithm """
num_hidden_layer = 4  # number of hidden layers for the networks
hidden_dim = 64  # dimension of hidden layers for the networks
num_workers = 2
net_list2 = []
for i in range(num_workers + 1):
    with tf.name_scope('A3C'):
        with tf.name_scope('Actor'):
            actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                            hidden_dim_list=num_hidden_layer * [hidden_dim])
        with tf.name_scope('Critic'):
            critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
    net_list = [actor, critic]
    net_list2.append(net_list)

""" choose optimizers """
actor_lr, critic_lr = 5e-5, 1e-4  # learning rate
a_optimizer = tf.optimizers.RMSprop(actor_lr)
c_optimizer = tf.optimizers.RMSprop(critic_lr)
optimizers_list = [a_optimizer, c_optimizer]

model = A3C(net_list2, optimizers_list, entropy_beta=0.005)
""" 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
entropy_beta: factor for entropy boosted exploration
"""

env_list = []
for i in range(num_workers):
    env_list.append(gym.make(env_id).unwrapped)
model.learn(env_list, train_episodes=20000, test_episodes=100, max_steps=20000, n_workers=num_workers, update_itr=10,
            gamma=0.99, save_interval=500, mode='train')
""" 
full list of parameters for training
---------------------------------------
env_list: a list of same learning environments
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
n_workers: manually set number of workers
update_itr: update global policy after several episodes
gamma: reward discount factor
save_interval: timesteps for saving the weights and plotting the results
mode: train or test
"""
# test
model.learn(env_list, test_episodes=100, max_steps=20000, mode='test', render=True)
