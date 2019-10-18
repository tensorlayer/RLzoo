from common.env_wrappers import *
from common.utils import *
from algorithms import *

# EnvName = 'CartPole-v1'
# EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'dm_control'][0]

# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'DQN')
# alg = DQN('train', **alg_params)
# alg.learn(env=env, number_timesteps=int(1e4), save_path=None,
#           save_interval=0, **learn_params)


# # EnvName = 'Pendulum-v0'
# EnvName = 'FetchReach-v1'
# EnvName = 'Ant-v2' 
# EnvName = 'CartPole-v0'  # classic_control, ac cannot learn cartpole-v1
# EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'robotics', 'dm_control'][0]
# EnvName = 'ReachTarget'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]

# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'AC')
# alg = AC(**alg_params)
# alg.learn(env=env, train_episodes=1000, test_episodes=1000, 
#         save_interval=100, mode='train', render=False, **learn_params)


# EnvName = 'Ant-v2'  # mujoco
# EnvName = 'FetchReach-v1'  # robotics
# EnvName = 'BipedalWalker-v2'  # box2d
# EnvName = 'CartPole-v1'  # classic_control
# EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'robotics', 'dm_control'][0]
# EnvName = 'ToiletSeatUp'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]

# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'PG')
# alg = PG(**alg_params)
# alg.learn(env=env, train_episodes=1000, test_episodes=1000, 
#         save_interval=100, mode='train', render=False, **learn_params)


# EnvName = 'ToiletSeatUp'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]

# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, 'SAC')
# alg = SAC(**alg_params)
# alg.learn(env=env, train_episodes=1000, test_episodes=1000, 
#         save_interval=100, mode='train', render=False, **learn_params)


EnvName = 'Pendulum-v0'
EnvType = ['classic_control', 'atari', 'box2d', 'mujoco', 'dm_control'][0]
# EnvName = 'ToiletSea2tUp'
# EnvType = ['atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'][-1]

env = build_env(EnvName, EnvType)
alg_params, learn_params = call_default_params(env, EnvType, 'TD3')
alg = TD3(**alg_params)
alg.learn(env=env, train_episodes=1000, test_episodes=1000, 
        save_interval=100, mode='train', render=False, **learn_params)

