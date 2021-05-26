import multiprocessing as mp
from functools import partial


def build_network(observation_space, action_space, name='DPPO_CLIP'):
    """ build networks for the algorithm """
    from rlzoo.common.policy_networks import StochasticPolicyNetwork
    from rlzoo.common.value_networks import ValueNetwork

    hidden_dim = 64
    num_hidden_layer = 2
    critic = ValueNetwork(observation_space, [hidden_dim] * num_hidden_layer, name=name + '_value')

    actor = StochasticPolicyNetwork(observation_space, action_space,
                                    [hidden_dim] * num_hidden_layer,
                                    trainable=True,
                                    name=name + '_policy')
    return critic, actor


def build_opt(actor_lr=1e-4, critic_lr=2e-4):
    import tensorflow as tf
    return [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]


def make_sampler_process(num, create_env_func, should_stop):
    from dppo_clip_distributed.dppo_sampler import DPPOSampler
    process_list = []
    sample_pipe_list = []
    for _ in range(num):
        sampler = DPPOSampler(create_env_func)
        pipe_a, pipe_b = mp.Pipe()
        p = mp.Process(target=sampler.run, args=(pipe_a, should_stop))
        p.daemon = True
        process_list.append(p)
        sample_pipe_list.append(pipe_b)
    return process_list, sample_pipe_list


def make_infer_server_process(
        build_net_func, sample_pipe_list, traj_queue, should_stop, should_update, barrier,
        net_param_pipe):
    from dppo_clip_distributed.dppo_infer_server import DPPOInferServer
    build_net_func = partial(build_net_func, 'DPPO_CLIP_INFER_SERVER')

    infer_server = DPPOInferServer(build_net_func)
    p = mp.Process(
        target=infer_server.run,
        args=(sample_pipe_list, traj_queue, should_stop, should_update, barrier, net_param_pipe))
    p.daemon = True
    return [p]


def make_learner_process(
        num, build_net_func, traj_queue, grad_queue, should_stop_event, should_update_event, barrier,
        param_que_list):
    process_list = []
    l = len(str(num))

    from dppo_clip_distributed.dppo_learner import DPPOLearner

    for i in range(num):
        net_func = partial(build_net_func, 'DPPO_CLIP_LEARNER_{}'.format(str(i).zfill(l)))
        learner = DPPOLearner(net_func)
        p = mp.Process(
            target=learner.run,
            args=(traj_queue, grad_queue, should_stop_event, should_update_event, barrier, param_que_list[i]))
        p.daemon = True
        process_list.append(p)
    return process_list


def make_global_manager(
        build_net_func, build_opt_func, traj_queue, grad_queue, should_stop, should_update, barrier, param_que_list):
    from dppo_clip_distributed.dppo_global_manager import DPPOGlobalManager
    build_net_func = partial(build_net_func, 'DPPO_CLIP_GLOBAL')
    global_manager = DPPOGlobalManager(build_net_func, build_opt_func)

    p = mp.Process(
        target=global_manager.run,
        args=(traj_queue, grad_queue, should_stop, should_update, barrier, param_que_list))
    p.daemon = True
    return [p]

