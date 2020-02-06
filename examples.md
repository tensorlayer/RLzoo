# Examples

## Descriptions of Algorithms and Environments in RLZoo

| Algorithms                 | Action Space        | Policy        | Update     | Envs                                                         |
| -------------------------- | ------------------- | ------------- | ---------- | ------------------------------------------------------------ |
| DQN (double, dueling, PER) | Discrete Only       | --            | Off-policy | Atari, Classic Control                                       |
| AC                         | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |
| PG                         | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |
| DDPG                       | Continuous          | Deterministic | Off-policy | Classic Control, Box2D, Mujoco, Robotics, DeepMind Control, RLBench |
| TD3                        | Continuous          | Deterministic | Off-policy | Classic Control, Box2D, Mujoco, Robotics, DeepMind Control, RLBench |
| SAC                        | Continuous          | Stochastic    | Off-policy | Classic Control, Box2D, Mujoco, Robotics, DeepMind Control, RLBench |
| A3C                        | Discrete/Continuous | Stochastic    | On-policy  | Atari, Classic Control, Box2D, Mujoco, Robotics, DeepMind Control |
| PPO                        | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |
| DPPO                       | Discrete/Continuous | Stochastic    | On-policy  | Atari, Classic Control, Box2D, Mujoco, Robotics, DeepMind Control |
| TRPO                       | Discrete/Continuous | Stochastic    | On-policy  | All                                                          |



## 1. Deep Q-Network (DQN)

```python
AlgName = 'DQN'
EnvName = 'PongNoFrameskip-v4'
EnvType = 'atari'
# EnvName = 'CartPole-v1'
# EnvType = 'classic_control'  # the name of env needs to match the type of env

env = build_env(EnvName, EnvType)
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)

```

## 2. Actor-Critic (AC)

```python
AlgName = 'AC'
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
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)

```

## 3. Policy Gradient (PG)

```python
AlgName = 'PG'
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
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)
```

## 4. Deep Deterministic Policy Gradient (DDPG)

```python
AlgName = 'DDPG'
EnvName = 'Pendulum-v0'  # only continuous action
EnvType = 'classic_control'

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
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)

```



## 5. Twin Delayed DDPG (TD3)

```python
AlgName = 'TD3'
EnvName = 'Pendulum-v0'  # only continuous action
EnvType = 'classic_control'

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
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)
```

## 6. Soft Actor-Critic (SAC)

```python
AlgName = 'SAC'
EnvName = 'Pendulum-v0'  # only continuous action
EnvType = 'classic_control'

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
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)
```

## 7. Asynchronous Advantage Actor-Critic (A3C)

```python
AlgName = 'A3C'
EnvName = 'PongNoFrameskip-v4'
EnvType = 'atari'

# EnvName = 'Pendulum-v0'  # only continuous action
# EnvType = 'classic_control'

# EnvName = 'BipedalWalker-v2'
# EnvType = 'box2d'

# EnvName = 'Ant-v2'
# EnvType = 'mujoco'

# EnvName = 'FetchPush-v1'
# EnvType = 'robotics'

# EnvName = 'FishSwim-v0'
# EnvType = 'dm_control'

number_workers = 2  # need to specify number of parallel workers
env = build_env(EnvName, EnvType, nenv=number_workers)
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env,  mode='train', render=False, **learn_params)
alg.learn(env=env,  mode='test', render=True, **learn_params)
```

## 8. Proximal Policy Optimization (PPO)

```python
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
alg_params, learn_params = call_default_params(env, EnvType, 'PPO')
alg = PPO(method='clip', **alg_params) # specify 'clip' or 'penalty' method for PPO
alg.learn(env=env,  mode='train', render=False, **learn_params)
alg.learn(env=env,  mode='test', render=False, **learn_params)
```

## 9. Distributed Proximal Policy Optimization (DPPO)

```python
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

number_workers = 2  # need to specify number of parallel workers
env = build_env(EnvName, EnvType, nenv=number_workers)
alg_params, learn_params = call_default_params(env, EnvType, 'DPPO')
alg = DPPO(method='penalty', **alg_params) # specify 'clip' or 'penalty' method for PPO
alg.learn(env=env,  mode='train', render=False, **learn_params)
alg.learn(env=env,  mode='test', render=True, **learn_params)
```

## 10. Trust Region Policy Optimization (TRPO)

```python
AlgName = 'TRPO'
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
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)
```

