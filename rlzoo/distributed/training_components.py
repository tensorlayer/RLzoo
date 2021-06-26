from rlzoo.common.env_wrappers import build_env
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
from rlzoo.algorithms.dppo_clip_distributed.dppo_clip import DPPO_CLIP
from functools import partial

# Specify the training configurations
training_conf = {
    'total_step': int(1e7),  # overall training timesteps
    'traj_len': 200,         # length of the rollout trajectory
    'train_n_traj': 2,       # update the models after every certain number of trajectories for each learner 
    'save_interval': 10,     # saving the models after every certain number of updates
}

# Specify the environment and launch it
env_name, env_type = 'CartPole-v0', 'classic_control'
env_maker = partial(build_env, env_name, env_type)
temp_env = env_maker()
obs_shape, act_shape = temp_env.observation_space.shape, temp_env.action_space.shape

env_conf = {
    'env_name': env_name,
    'env_type': env_type,
    'env_maker': env_maker,
    'obs_shape': obs_shape,
    'act_shape': act_shape,
}


def build_network(observation_space, action_space, name='DPPO_CLIP'):
    """ build networks for the algorithm """
    hidden_dim = 256
    num_hidden_layer = 2
    critic = ValueNetwork(observation_space, [hidden_dim] * num_hidden_layer, name=name + '_value')

    actor = StochasticPolicyNetwork(observation_space, action_space,
                                    [hidden_dim] * num_hidden_layer,
                                    trainable=True,
                                    name=name + '_policy')
    return critic, actor


def build_opt(actor_lr=1e-4, critic_lr=2e-4):
    """ choose the optimizer for learning """
    import tensorflow as tf
    return [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]


net_builder = partial(build_network, temp_env.observation_space, temp_env.action_space)
opt_builder = partial(build_opt, )

agent_conf = {
    'net_builder': net_builder,
    'opt_builder': opt_builder,
    'agent_generator': partial(DPPO_CLIP, net_builder, opt_builder),
}
del temp_env

from rlzoo.distributed.start_dis_role import main

print('Start Training.')
main(training_conf, env_conf, agent_conf)
print('Training Finished.')
