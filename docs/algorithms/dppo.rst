DPPO
===========================

Example
-----------

.. code-block:: python
   :linenos:

    from rlzoo.common.env_wrappers import build_env
    from rlzoo.common.utils import call_default_params
    from rlzoo.algorithms import DPPO

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

Distributed Proximal Policy Optimization (Penalty)
----------------------------------------------------

.. autoclass:: rlzoo.algorithms.dppo_penalty.dppo_penalty.DPPO_PENALTY
   :members:
   :undoc-members:


Distributed Proximal Policy Optimization (Clip)
------------------------------------------------

.. autoclass:: rlzoo.algorithms.dppo_clip.dppo_clip.DPPO_CLIP
   :members:
   :undoc-members:

Default Hyper-parameters
----------------------------------

.. automodule:: rlzoo.algorithms.dppo.default
   :members:
   :undoc-members:
   :show-inheritance:

