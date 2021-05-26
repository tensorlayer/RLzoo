from dppo_clip_distributed.dppo_roles_func import *
from rlzoo.common.env_wrappers import build_env

if __name__ == '__main__':

    n_sampler = 3
    name = 'DPPO_CLIP'
    build_env = partial(build_env, 'CartPole-v0', 'classic_control')
    env = build_env()
    observation_space, action_space = env.observation_space, env.action_space
    build_network = partial(build_network, observation_space, action_space)

    traj_queue = mp.Queue(maxsize=10000)
    should_stop_event = mp.Event()
    should_stop_event.clear()
    should_update_event = mp.Event()
    should_update_event.set()
    grad_queue = mp.Queue(maxsize=10000), mp.Queue(maxsize=10000),

    n_learner = 2
    update_barrier = mp.Barrier(2 + n_learner)  # InferServer + Learner + GlobalManager
    param_que_list = [mp.Queue() for _ in range(1 + n_learner)]  # InferServer + Learner

    process_list = []

    p_list, sample_pipe_list = make_sampler_process(
        n_sampler, build_env, should_stop_event
    )
    process_list.extend(p_list)

    p_list = make_infer_server_process(
        build_network, sample_pipe_list, traj_queue, should_stop_event, should_update_event, update_barrier,
        param_que_list[0])
    process_list.extend(p_list)

    p_list = make_learner_process(
        n_learner, build_network, traj_queue, grad_queue, should_stop_event, should_update_event, update_barrier,
        param_que_list[1:])
    process_list.extend(p_list)

    if True:
        for p in process_list:
            p.start()
        from dppo_clip_distributed.dppo_global_manager import DPPOGlobalManager
        build_net_func = partial(build_network, 'DPPO_CLIP_GLOBAL')
        global_manager = DPPOGlobalManager(build_net_func, build_opt)

        global_manager.run(traj_queue, grad_queue, should_stop_event, should_update_event, update_barrier, param_que_list)
    else:
        import time
        p_list = make_global_manager(
            build_network, build_opt, traj_queue, grad_queue, should_stop_event, should_update_event, update_barrier,
            param_que_list)
        process_list.extend(p_list)
        for p in process_list:
            p.start()
        while True:
            print('traj_queue {} grad_queue {}'.format(traj_queue.qsize(), grad_queue[0].qsize()))
            time.sleep(1)
