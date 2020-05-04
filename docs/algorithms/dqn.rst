DQN and Variants
=================================

Example
------------

.. code-block:: python
   :linenos:

    from rlzoo.common.env_wrappers import build_env
    from rlzoo.common.utils import call_default_params
    from rlzoo.algorithms import DQN
   
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

Deep Q-Networks
---------------------------------

.. autoclass:: rlzoo.algorithms.dqn.dqn.DQN
   :members:
   :undoc-members:

Default Hyper-parameters
----------------------------------

.. automodule:: rlzoo.algorithms.dqn.default
   :members:
   :undoc-members:
   :show-inheritance:

