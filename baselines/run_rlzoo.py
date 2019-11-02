from common.env_wrappers import *
from common.utils import *
from algorithms import *

EnvName = 'PongNoFrameskip-v4'
EnvType = 'atari'

# EnvName = 'CartPole-v0'
# EnvType = 'classic_control'

# EnvName = 'BipedalWalker-v2'
# EnvType = 'box2d'

# EnvName = 'Ant-v2'
# EnvType = 'mujoco'

# EnvName = 'FetchPush-v1'
# EnvType = 'robotics'

# EnvName = 'FishSwim-v0'
# EnvType = 'dm_control'

# EnvName = 'ReachTarget'
# EnvType = 'rlbench'

env = build_env(EnvName, EnvType)
alg_params, learn_params = call_default_params(env, EnvType, 'SAC')
alg = SAC(**alg_params)
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)


# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'DPPO')
# alg = DPPO(method='penalty', **alg_params) # specify 'clip' or 'penalty' method for different version of PPO
# alg.learn(env=env,  mode='train', render=False, **learn_params)
# alg.learn(env=env,  mode='test', render=False, **learn_params)


# number_workers = 2
# env = build_env(EnvName, EnvType, nenv=number_workers)
# alg_params, learn_params = call_default_params(env, EnvType, 'A3C')
# alg = A3C(**alg_params)
# alg.learn(env=env,  mode='train', n_workers=number_workers, render=False, **learn_params)
# alg.learn(env=env,  mode='test', n_workers=number_workers, render=True, **learn_params)



