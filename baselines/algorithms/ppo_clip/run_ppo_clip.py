import gym

# from common.env_wrappers import DummyVecEnv
from common.utils import make_env
from algorithms.ppo_clip.ppo_clip import PPO_CLIP
from common.value_networks import *
from common.policy_networks import *


''' load environment '''
env = gym.make('Pendulum-v0').unwrapped

state_dim = env.observation_space.shape
action_dim = env.action_space.shape
action_max = env.action_space.high
action_min = env.action_space.low


''' build networks for the algorithm '''
name = 'ppo_penalty'
hidden_dim = 100
num_hidden_layer = 1
critic = MlpValueNetwork(state_dim, [hidden_dim] * num_hidden_layer, name=name + '_value')

actor = StochasticPolicyNetwork(state_dim, action_dim, [hidden_dim] * num_hidden_layer, trainable=True,
                                name=name + '_policy')
actor_old = StochasticPolicyNetwork(state_dim, action_dim, [hidden_dim] * num_hidden_layer, trainable=False,
                                    name=name + '_old_policy')
net_list = critic, actor, actor_old

''' create model '''
actor_lr = 1e-4
critic_lr = 2e-4
optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]

model = PPO_CLIP(net_list, optimizers_list, action_dim[0], state_dim[0], [action_min, action_max])
'''
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
s_dim: dimension of action for the environment
a_dim: dimension of state for the environment
a_bounds: a list of [min_action, max_action] action bounds for the environment
epsilon: clip parameter
'''

model.learn(env, reward_shaping=lambda x: (x + 8) / 8)
'''
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps: maximum number of steps for one episode
save_interval: timesteps for saving
gamma: reward discount factor
mode: train or test
render: render each step
batch_size: udpate batchsize
a_update_steps: actor update iteration steps
c_update_steps: critic update iteration steps
seed: random seed
reward_shaping: reward shaping function
:return: None
'''

obs = env.reset()
for i in range(1000):
    action = model.get_action(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

env.close()
