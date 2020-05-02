Quick Start
=================================

Open ``./run_rlzoo.py``:

.. code-block:: python
   :linenos:

    from rlzoo.common.env_wrappers import build_env
    from rlzoo.common.utils import call_default_params
    from rlzoo.algorithms import TD3
    # choose an algorithm
    AlgName = 'TD3'
    # chose an environment
    EnvName = 'Pendulum-v0'  
    # select a corresponding environment type
    EnvType = 'classic_control'
    # build an environment with wrappers
    env = build_env(EnvName, EnvType)  
    # call default parameters for the algorithm and learning process
    alg_params, learn_params = call_default_params(env, EnvType, AlgName)  
    # instantiate the algorithm
    alg = eval(AlgName+'(**alg_params)')
    # start the training
    alg.learn(env=env, mode='train', render=False, **learn_params)  
    # test after training 
    alg.learn(env=env, mode='test', render=True, **learn_params)  


Run the example:

.. code-block:: bash
   :linenos:

   python run_rlzoo.py
