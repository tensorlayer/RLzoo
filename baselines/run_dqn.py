import gym
from baselines.algorithms.dqn import dqn, default
from baselines.common.env_wrappers import *


env = gym.make('CartPole-v1')
env = Monitor(env)
alg = dqn.DQN('train')
params = default.classic_control(env)
alg.learn(env=env, number_timesteps=int(5e4), save_path=None,
          save_interval=0, **params)


env = gym.make('PongNoFrameskip-v4')
# wrap the env
env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env, skip=4)
env = Monitor(env)
env = EpisodicLifeEnv(env)
if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
env = WarpFrame(env)
env = ClipRewardEnv(env)
env = FrameStack(env, 4)

# init dqn algorithm
alg = dqn.DQN('train')
params = default.atari(env)
alg.learn(env=env, number_timesteps=int(5e4), save_path=None,
          save_interval=0, **params)
