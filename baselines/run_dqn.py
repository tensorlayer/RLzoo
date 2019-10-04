from algorithms.dqn import dqn, default
from common.env_wrappers import *


env = build_env('CartPole-v1', 'classic_control')
alg = dqn.DQN('train')
params = default.classic_control(env)
alg.learn(env=env, number_timesteps=int(1e4), save_path=None,
          save_interval=0, **params)


env = build_env('PongNoFrameskip-v4', 'atari')
alg = dqn.DQN('train')
params = default.atari(env)
alg.learn(env=env, number_timesteps=int(5e4), save_path=None,
          save_interval=0, **params)
