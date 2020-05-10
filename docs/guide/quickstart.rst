Quick Start
=================================

Simple Usage
---------------

Open ``./run_rlzoo.py``:

.. code-block:: python
   :linenos:

    from rlzoo.common.env_wrappers import build_env
    from rlzoo.common.utils import call_default_params
    from rlzoo.algorithms import TD3
    # choose an algorithm
    AlgName = 'TD3' 
    # select a corresponding environment type
    EnvType = 'classic_control'
    # chose an environment
    EnvName = 'Pendulum-v0' 
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

   python run_rlzoo.py


Choices for ``AlgName``: 'DQN', 'AC', 'A3C', 'DDPG', 'TD3', 'SAC', 'PG', 'TRPO', 'PPO', 'DPPO'

Choices for ``EnvType``: 'atari', 'box2d', 'classic_control', 'mujoco', 'robotics', 'dm_control', 'rlbench'

Choices for ``EnvName`` refers to :ref:`env_list`


Another Usage
---------------

For providing more flexibility, we provide another usage example of RLzoo with more explicit configurations as follows, where the users can pass in customized networks and otpimizers, etc.

.. code-block:: python
   :linenos:

    import gym
    from rlzoo.common.utils import make_env, set_seed
    from rlzoo.algorithms import AC
    from rlzoo.common.value_networks import ValueNetwork
    from rlzoo.common.policy_networks import StochasticPolicyNetwork

    ''' load environment '''
    env = gym.make('CartPole-v0').unwrapped
    obs_space = env.observation_space
    act_space = env.action_space
    # reproducible
    seed = 2
    set_seed(seed, env)

    ''' build networks for the algorithm '''
    num_hidden_layer = 4 #number of hidden layers for the networks
    hidden_dim = 64 # dimension of hidden layers for the networks
    with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                    # choose the critic network, can be replaced with customized network
                    critic = ValueNetwork(obs_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                    # choose the actor network, can be replaced with customized network
                    actor = StochasticPolicyNetwork(obs_space, act_space, hidden_dim_list=num_hidden_layer * [hidden_dim], output_activation=tf.nn.tanh)
    net_list = [actor, critic] # list of the networks

    ''' choose optimizers '''
    a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
    a_optimizer = tf.optimizers.Adam(a_lr)
    c_optimizer = tf.optimizers.Adam(c_lr)
    optimizers_list=[a_optimizer, c_optimizer]  # list of optimizers

    # intialize the algorithm model, with algorithm parameters passed in
    model = AC(net_list, optimizers_list)
    ''' 
    full list of arguments for the algorithm
    ----------------------------------------
    net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
    optimizers_list: a list of optimizers for all networks and differentiable variables
    gamma: discounted factor of reward
    action_range: scale of action values
    '''

    # start the training process, with learning parameters passed in
    model.learn(env, train_episodes=500,  max_steps=200,
                save_interval=50, mode='train', render=False)
    ''' 
    full list of parameters for training
    -------------------------------------
    env: learning environment
    train_episodes:  total number of episodes for training
    test_episodes:  total number of episodes for testing
    max_steps:  maximum number of steps for one episode
    save_interval: time steps for saving the weights and plotting the results
    mode: 'train' or 'test'
    render:  if true, visualize the environment
    '''

    # test after training
    model.learn(env, test_episodes=100, max_steps=200,  mode='test', render=True)



Interactive Configurations
--------------------------

We also provide an interactive learning configuration with Jupyter Notebook and *ipywidgets*, where you can select the algorithm, environment, and general learning settings with simple clicking on dropdown lists and sliders! 
A video demonstrating the usage is as following. 
The interactive mode can be used with `rlzoo/interactive/main.ipynb <https://github.com/tensorlayer/RLzoo/blob/master/rlzoo/interactive/main.ipynb>`_ by running ``$ jupyter notebook`` to open it.

.. image:: ../../gif/interactive.gif