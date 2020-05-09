from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
from rlzoo.common.utils import set_seed

""" 
full list of algorithm parameters (alg_params)
-----------------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
entropy_beta: factor for entropy boosted exploration
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
n_workers: manually set number of workers
update_itr: update global policy after several episodes
gamma: reward discount factor
save_interval: timesteps for saving the weights and plotting the results
mode: train or test
------------------------------------------------
"""


def atari(env, default_seed=True):
    if default_seed:
        assert isinstance(env, list)
        seed = np.arange(len(env)).tolist()  # a list of seeds for each env
        set_seed(seed, env)  # reproducible

    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict(
        entropy_beta=0.005
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env + 1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                    critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
                with tf.name_scope('Actor'):
                    actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                    hidden_dim_list=num_hidden_layer * [hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.RMSprop(a_lr, name='RMS_optimizer_actor')
        c_optimizer = tf.optimizers.RMSprop(c_lr, name='RMS_optimizer_critic')
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=100,
        gamma=0.9,
        train_episodes=1000,
        test_episodes=10,
        save_interval=100,
        update_itr=10,
        n_workers=num_env
    )

    return alg_params, learn_params


def classic_control(env, default_seed=True):
    if default_seed:
        assert isinstance(env, list)
        seed = np.arange(len(env)).tolist()  # a list of seeds for each env
        set_seed(seed, env)  # reproducible

    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict(
        entropy_beta=0.005
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env + 1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                    critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
                with tf.name_scope('Actor'):
                    actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                    hidden_dim_list=num_hidden_layer * [hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.RMSprop(a_lr, name='RMS_optimizer_actor')
        c_optimizer = tf.optimizers.RMSprop(c_lr, name='RMS_optimizer_critic')
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=100,
        gamma=0.9,
        train_episodes=1000,
        test_episodes=10,
        save_interval=100,
        update_itr=10,
        n_workers=num_env
    )

    return alg_params, learn_params


def box2d(env, default_seed=True):
    if default_seed:
        assert isinstance(env, list)
        seed = np.arange(len(env)).tolist()  # a list of seeds for each env
        set_seed(seed, env)  # reproducible

    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict(
        entropy_beta=0.005
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env + 1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                    critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
                with tf.name_scope('Actor'):
                    actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                    hidden_dim_list=num_hidden_layer * [hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.RMSprop(a_lr, name='RMS_optimizer_actor')
        c_optimizer = tf.optimizers.RMSprop(c_lr, name='RMS_optimizer_critic')
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=20000,
        gamma=0.9,
        train_episodes=20000,
        test_episodes=10,
        save_interval=500,
        update_itr=10,
        n_workers=num_env
    )

    return alg_params, learn_params


def mujoco(env, default_seed=True):
    if default_seed:
        assert isinstance(env, list)
        seed = np.arange(len(env)).tolist()  # a list of seeds for each env
        set_seed(seed, env)  # reproducible

    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict(
        entropy_beta=0.005
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env + 1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                    critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
                with tf.name_scope('Actor'):
                    actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                    hidden_dim_list=num_hidden_layer * [hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.RMSprop(a_lr, name='RMS_optimizer_actor')
        c_optimizer = tf.optimizers.RMSprop(c_lr, name='RMS_optimizer_critic')
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=100,
        gamma=0.9,
        train_episodes=1000,
        test_episodes=10,
        save_interval=100,
        update_itr=10,
        n_workers=num_env
    )

    return alg_params, learn_params


def robotics(env, default_seed=True):
    if default_seed:
        assert isinstance(env, list)
        seed = np.arange(len(env)).tolist()  # a list of seeds for each env
        set_seed(seed, env)  # reproducible

    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict(
        entropy_beta=0.005
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env + 1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                    critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
                with tf.name_scope('Actor'):
                    actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                    hidden_dim_list=num_hidden_layer * [hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.RMSprop(a_lr, name='RMS_optimizer_actor')
        c_optimizer = tf.optimizers.RMSprop(c_lr, name='RMS_optimizer_critic')
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=100,
        gamma=0.9,
        train_episodes=1000,
        test_episodes=10,
        save_interval=100,
        update_itr=10,
        n_workers=num_env

    )

    return alg_params, learn_params


def dm_control(env, default_seed=True):
    if default_seed:
        assert isinstance(env, list)
        seed = np.arange(len(env)).tolist()  # a list of seeds for each env
        set_seed(seed, env)  # reproducible

    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict(
        entropy_beta=0.005
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env + 1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                    critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
                with tf.name_scope('Actor'):
                    actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                    hidden_dim_list=num_hidden_layer * [hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.RMSprop(a_lr, name='RMS_optimizer_actor')
        c_optimizer = tf.optimizers.RMSprop(c_lr, name='RMS_optimizer_critic')
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=100,
        gamma=0.9,
        train_episodes=1000,
        test_episodes=10,
        save_interval=100,
        update_itr=10,
        n_workers=num_env

    )

    return alg_params, learn_params


def rlbench(env, default_seed=True):
    if default_seed:
        assert isinstance(env, list)
        seed = np.arange(len(env)).tolist()  # a list of seeds for each env
        set_seed(seed, env)  # reproducible

    # for multi-threading
    if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
        num_env = len(env)  # number of envs passed in
        env = env[0]  # take one of the env as they are all the same
    else:
        num_env = 1

    alg_params = dict(
        entropy_beta=0.005
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 4  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        net_list2 = []  # networks list of networks list, each item for single thread/process
        for _ in range(num_env + 1):  # additional one for global
            with tf.name_scope('AC'):
                with tf.name_scope('Critic'):
                    critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
                with tf.name_scope('Actor'):
                    actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                    hidden_dim_list=num_hidden_layer * [hidden_dim])
            net_list = [actor, critic]
            net_list2.append(net_list)
        alg_params['net_list'] = net_list2
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-3, 1e-3  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.RMSprop(a_lr, name='RMS_optimizer_actor')
        c_optimizer = tf.optimizers.RMSprop(c_lr, name='RMS_optimizer_critic')
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        max_steps=100,
        gamma=0.9,
        train_episodes=1000,
        test_episodes=10,
        save_interval=100,
        update_itr=10,
        n_workers=num_env

    )

    return alg_params, learn_params