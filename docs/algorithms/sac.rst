SAC
===========================

Example
-----------

.. code-block:: python
   :linenos:

    from rlzoo.common.env_wrappers import build_env
    from rlzoo.common.utils import call_default_params
    from rlzoo.algorithms import SAC
   
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

Soft Actor-Critic
---------------------------------

.. autoclass:: rlzoo.algorithms.sac.sac.SAC
   :members:
   :undoc-members:

Default Hyper-parameters
----------------------------------

.. automodule:: rlzoo.algorithms.sac.default
   :members:
   :undoc-members:
   :show-inheritance:

