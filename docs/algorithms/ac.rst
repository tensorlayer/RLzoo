AC
===========================

Example
-----------

.. code-block:: python
   :linenos:

    from rlzoo.common.env_wrappers import build_env
    from rlzoo.common.utils import call_default_params
    from rlzoo.algorithms import AC
   
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

Actor-Critic
---------------------------------

.. autoclass:: rlzoo.algorithms.ac.ac.AC
   :members:
   :undoc-members:

Default Hyper-parameters
----------------------------------

.. automodule:: rlzoo.algorithms.ac.default
   :members:
   :undoc-members:
   :show-inheritance:

