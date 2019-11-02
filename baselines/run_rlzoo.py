from common.env_wrappers import *
from common.utils import *
from algorithms import *

EnvName = 'PongNoFrameskip-v4'
EnvType = 'atari'

# EnvName = 'Pendulum-v0' 
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
alg_params, learn_params = call_default_params(env, EnvType, 'DPPO')
alg = DPPO(method='clip', **alg_params) # specify 'clip' or 'penalty' method for different version of PPO
alg.learn(env=env,  mode='train', render=False, **learn_params)
alg.learn(env=env,  mode='test', render=False, **learn_params)


# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'SAC')
# alg = SAC(**alg_params)
# alg.learn(env=env, mode='train', render=False, **learn_params)
# alg.learn(env=env, mode='test', render=True, **learn_params)


# EnvName = 'Ant-v2'  # mujoco
# EnvName = 'FetchReach-v1'  # robotics
# EnvName = 'BipedalWalker-v2'  # box2d

# EnvName = 'Pendulum-v0'
# EnvName = 'CartPole-v1'  # classic_control
# EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'robotics', 'dm_control'][0]
# EnvName = 'ToiletSeatUp'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]

# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'PG')
# alg = PG(**alg_params)
# alg.learn(env=env, mode='train', render=False, **learn_params)


# EnvName = 'ToiletSeatUp'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]
# # EnvName = 'Pendulum-v0'
# # EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'dm_control'][0]
# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'SAC')
# alg = SAC(**alg_params)
# alg.learn(env=env, mode='train', render=False, **learn_params)


# EnvName = 'Pendulum-v0'
# EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'dm_control'][0]
# EnvName = 'ToiletSeatUp'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]

# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'TD3')
# alg = TD3(**alg_params)
# alg.learn(env=env, mode='train', render=False, **learn_params)

# # EnvName = 'Pendulum-v0'
# EnvName = 'CartPole-v0'  # classic_control, ac cannot learn cartpole-v1
# EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'robotics', 'dm_control'][0]
# # EnvName = 'ReachTarget'
# # EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]
# number_workers = 2
# env = build_env(EnvName, EnvType, nenv=number_workers)
# alg_params, learn_params = call_default_params(env, EnvType, 'A3C')
# alg = A3C(**alg_params)
# alg.learn(env=env,  mode='train', n_worker=number_workers, render=False, **learn_params)


# EnvName = 'Pendulum-v0'
# EnvName = 'CartPole-v0'  # classic_control, ac cannot learn cartpole-v1
# EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'robotics', 'dm_control'][0]
# EnvName = 'ReachTarget'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]
# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'DPPO_CLIP')
# alg = DPPO_CLIP(**alg_params)
# alg.learn(env=env,  mode='train', render=False, **learn_params)


# EnvName = 'PongNoFrameskip-v4'
# EnvType = 'atari'
# EnvName = 'CartPole-v1'
# EnvType = 'classic_control'
# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'DQN')
# alg = DQN(**alg_params)
# alg.learn(env=env, mode='train', **learn_params)
# alg.learn(env=env, mode='test', render=True, **learn_params)
