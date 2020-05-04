A3C
=================================

Example
-----------

.. code-block:: python
   :linenos:

    from rlzoo.common.env_wrappers import build_env
    from rlzoo.common.utils import call_default_params
    from rlzoo.algorithms import A3C

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
    
Asychronous Advantage Actor-Critic
----------------------------------------

.. autoclass:: rlzoo.algorithms.a3c.a3c.A3C
   :members:
   :undoc-members:

Default Hyper-parameters
----------------------------------

.. automodule:: rlzoo.algorithms.a3c.default
   :members:
   :undoc-members:
   :show-inheritance:
