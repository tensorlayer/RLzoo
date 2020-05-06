API
=================================

make_env()
----------------------

It can be used as:

.. code-block:: python
   :linenos:

    env = build_env(EnvName, EnvType)

call_default_params()
----------------------

It can be used as:

.. code-block:: python
   :linenos:

    alg_params, learn_params = call_default_params(env, EnvType, AlgName)

The ``call_default_params`` returns the hyper-parameters stored in two dictionaries ``alg_params`` and ``learn_params``, which can be printed to see what are contained inside. Hyper-parameters in these two dictionaries can also be changed by users before instantiating the agent and starting the learning process.

If you want to know exactly where the default hyper-parameters come from, they are stored in an individual Python script as ``default.py`` in each algorithm file in ``./rlzoo/algorithms/``.

alg.learn()
------------

It can be used as:

.. code-block:: python
   :linenos:

    # start the training
    alg.learn(env=env, mode='train', render=False, **learn_params)
    # test after training
    alg.learn(env=env, mode='test', render=True, **learn_params)

where the ``alg`` is an instantiation of DRL algorithm in RLzoo.