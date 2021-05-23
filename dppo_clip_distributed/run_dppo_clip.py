from rlzoo.common.utils import set_seed
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *
from rlzoo.common.env_wrappers import build_env
import multiprocessing as mp
from dppo_clip_distributed.dppo_infer_server import DPPOInferServer
from dppo_clip_distributed.dppo_learner import DPPOLearner
from dppo_clip_distributed.dppo_global_manager import DPPOGlobalManager
from dppo_clip_distributed.dppo_sampler import DPPOSampler
from functools import partial


def make_network(observation_space, action_space, name='DPPO_CLIP'):
    """ build networks for the algorithm """
    hidden_dim = 64
    num_hidden_layer = 2
    critic = ValueNetwork(observation_space, [hidden_dim] * num_hidden_layer, name=name + '_value')

    actor = StochasticPolicyNetwork(observation_space, action_space,
                                    [hidden_dim] * num_hidden_layer,
                                    trainable=True,
                                    name=name + '_policy')
    return critic, actor


def make_opt(actor_lr=1e-4, critic_lr=2e-4):
    return [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]


def make_sampler_process(num, create_env_func, should_stop):
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
        build_net_func, create_env_func, sample_pipe_list, traj_queue, should_stop, should_update, barrier,
        net_param_pipe):
    infer_server = DPPOInferServer(build_net_func, net_param_pipe)
    p = mp.Process(target=infer_server.run,
                   args=(create_env_func, sample_pipe_list, traj_queue, should_stop, should_update, barrier))
    p.daemon = True
    return [infer_server]


def make_learner_process(
        num, build_net_func, traj_queue, grad_queue, should_stop_event, should_update_event, barrier, net_param_pipe):
    process_list = []
    for _ in range(num):
        learner = DPPOLearner(build_net_func, net_param_pipe)
        p = mp.Process(
            target=learner.run, args=(traj_queue, grad_queue, should_stop_event, should_update_event, barrier))
        p.daemon = True
        process_list.append(p)
    return process_list


def make_global_manager(
        build_net_func, build_opt_func, n_nets, traj_queue, grad_queue, should_stop, should_update, barrier, ):
    param_pipe_a, param_pipe_b = zip(*[mp.Pipe() for _ in range(n_nets)])
    global_manager = DPPOGlobalManager(build_net_func, build_opt_func, param_pipe_a)
    p = mp.Process(target=global_manager.run, args=(traj_queue, grad_queue, should_stop, should_update, barrier,))
    p.daemon = True
    return [p], param_pipe_b


