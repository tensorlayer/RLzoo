from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
from rlzoo.common.utils import set_seed

""" 
full list of algorithm parameters (alg_params)
-----------------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
epsilon: clip parameter  (for method 'clip')
kl_target: controls bounds of policy update and adaptive lambda  (for method 'penalty')
lam:  KL-regularization coefficient   (for method 'penalty')
-----------------------------------------------

full list of learning parameters (learn_params)
-----------------------------------------------
train_episodes: total number of episodes for training
test_episodes: total number of episodes for testing
max_steps:  maximum number of steps for one episode
save_interval: time steps for saving
gamma: reward discount factor
mode: train or test
batch_size: update batch size
a_update_steps: actor update iteration steps
c_update_steps: critic update iteration steps
n_worker: number of workers
-----------------------------------------------
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

    alg_params = dict(method='clip',  # method can be clip or penalty
                      epsilon=0.2,  # for method 'clip'
                      kl_target=0.01,  # for method 'penalty'
                      lam=0.5  # for method 'penalty'
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     [hidden_dim] * num_hidden_layer)

        net_list = v_net, policy_net
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=1000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=50,
                        gamma=0.9,
                        a_update_steps=10,
                        c_update_steps=10,
                        n_workers=num_env,
                        batch_size=32)

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

    alg_params = dict(method='clip',  # method can be clip or penalty
                      epsilon=0.2,  # for method 'clip'
                      kl_target=0.01,  # for method 'penalty'
                      lam=0.5  # for method 'penalty'
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     [hidden_dim] * num_hidden_layer)

        net_list = v_net, policy_net
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=1000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=50,
                        gamma=0.9,
                        a_update_steps=10,
                        c_update_steps=10,
                        n_workers=num_env,
                        batch_size=32)

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

    alg_params = dict(method='clip',  # method can be clip or penalty
                      epsilon=0.2,  # for method 'clip'
                      kl_target=0.01,  # for method 'penalty'
                      lam=0.5  # for method 'penalty'
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     [hidden_dim] * num_hidden_layer)

        net_list = v_net, policy_net
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=1000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=50,
                        gamma=0.9,
                        a_update_steps=10,
                        c_update_steps=10,
                        n_workers=num_env,
                        batch_size=32)

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

    alg_params = dict(method='clip',  # method can be clip or penalty
                      epsilon=0.2,  # for method 'clip'
                      kl_target=0.01,  # for method 'penalty'
                      lam=0.5  # for method 'penalty'
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     [hidden_dim] * num_hidden_layer)

        net_list = v_net, policy_net
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=1000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=50,
                        gamma=0.9,
                        a_update_steps=10,
                        c_update_steps=10,
                        n_workers=num_env,
                        batch_size=32)

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

    alg_params = dict(method='clip',  # method can be clip or penalty
                      epsilon=0.2,  # for method 'clip'
                      kl_target=0.01,  # for method 'penalty'
                      lam=0.5  # for method 'penalty'
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     [hidden_dim] * num_hidden_layer)

        net_list = v_net, policy_net
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=1000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=50,
                        gamma=0.9,
                        a_update_steps=10,
                        c_update_steps=10,
                        n_workers=num_env,
                        batch_size=32)

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

    alg_params = dict(method='clip',  # method can be clip or penalty
                      epsilon=0.2,  # for method 'clip'
                      kl_target=0.01,  # for method 'penalty'
                      lam=0.5  # for method 'penalty'
    )

    if alg_params.get('net_list') is None:
        num_hidden_layer = 2  # number of hidden layers for the networks
        hidden_dim = 64  # dimension of hidden layers for the networks
        with tf.name_scope('DPPO'):
            with tf.name_scope('V_Net'):
                v_net = ValueNetwork(env.observation_space, [hidden_dim] * num_hidden_layer)
            with tf.name_scope('Policy'):
                policy_net = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                     [hidden_dim] * num_hidden_layer)

        net_list = v_net, policy_net
        alg_params['net_list'] = net_list

    if alg_params.get('optimizers_list') is None:
        actor_lr = 1e-4
        critic_lr = 2e-4
        optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(train_episodes=1000,
                        test_episodes=100,
                        max_steps=200,
                        save_interval=50,
                        gamma=0.9,
                        a_update_steps=10,
                        c_update_steps=10,
                        n_workers=num_env,
                        batch_size=32)

    return alg_params, learn_params
