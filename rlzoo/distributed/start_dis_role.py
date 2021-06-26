import argparse

from rlzoo.distributed.dis_components import *
import tensorflow as tf
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-l', type=int, default=1)
    p.add_argument('-a', type=int, default=1)
    p.add_argument('-s', type=int, default=1)
    p.add_argument('-f', type=str, default='')  # config.json

    args = p.parse_args()
    return args


def run_learner(agent, args, training_conf, env_conf, agent_conf):
    agent_generator = agent_conf['agent_generator']
    total_step, traj_len, train_n_traj = training_conf['total_step'], training_conf['traj_len'], training_conf['train_n_traj'],
    obs_shape, act_shape = env_conf['obs_shape'], env_conf['act_shape']
    
    if agent.role_rank() == 0:
        param_q = agent.new_queue((Role.Learner, 0), (Role.Server, 0))

    traj_q = agent.new_queue((Role.Server, 0), (Role.Learner, agent.role_rank()))

    rl_agent = agent_generator()
    rl_agent.init_components()

    # init model
    rl_agent.update_model([agent.role_all_reduce(weights) for weights in rl_agent.all_weights])

    if agent.role_rank() == 0:
        for weight in rl_agent.all_weights:
            param_q.put(tf.Variable(weight, dtype=tf.float32))

    n_update = total_step // (traj_len * agent.role_size(Role.Learner) * train_n_traj)
    for i in range(n_update):
        traj_list = [[traj_q.get(dtype=tf.float32, shape=(traj_len, *shape)) for shape in [
            obs_shape, act_shape, (), (), obs_shape, (), (1,)]] for _ in range(train_n_traj)]

        rl_agent.train(traj_list, dis_agent=agent)

        # send weights to server
        if agent.role_rank() == 0:
            for weight in rl_agent.all_weights:
                param_q.put(tf.Variable(weight, dtype=tf.float32))
    print('learner finished')


def run_actor(agent, args, training_conf, env_conf):  # sampler
    env_maker, total_step = env_conf['env_maker'], training_conf['total_step']
    
    from gym import spaces

    env = env_maker()
    action_q, step_data_q = agent.new_queue_pair((Role.Server, 0), (Role.Actor, agent.role_rank()))

    state, reward, done = env.reset(), 0, False
    each_total_step = int(total_step/agent.role_size(Role.Actor))
    action_dtype = tf.int32 if isinstance(env.action_space, spaces.Discrete) else tf.float32
    for i in range(each_total_step):
        step_data_q.put(tf.Variable(state, dtype=tf.float32))
        a = action_q.get(dtype=action_dtype, shape=env.action_space.shape).numpy()
        next_state, reward, done, _ = env.step(a)
        for data in (reward, done, next_state):
            step_data_q.put(tf.Variable(data, dtype=tf.float32))
        if done:
            state = env.reset()
        else:
            state = next_state
    print('actor finished')


def run_server(agent, args, training_conf, env_conf, agent_conf):
    total_step, traj_len, train_n_traj, save_interval = training_conf['total_step'], training_conf['traj_len'], \
                                                        training_conf['train_n_traj'], training_conf['save_interval'],
    obs_shape, env_name = env_conf['obs_shape'], env_conf['env_name']
    agent_generator = agent_conf['agent_generator']

    from rlzoo.algorithms.dppo_clip_distributed.dppo_clip import DPPO_CLIP
    from rlzoo.distributed.dis_components import Role
    from gym import spaces

    learner_size = agent.role_size(Role.Learner)
    rl_agent: DPPO_CLIP = agent_generator()
    rl_agent.init_components()

    # queue to actor
    q_list = [agent.new_queue_pair((Role.Server, 0), (Role.Actor, i)) for i in
              range(agent.role_size(Role.Actor))]
    action_q_list, step_data_q_list = zip(*q_list)

    # queue to learner
    param_q = agent.new_queue((Role.Learner, 0), (Role.Server, 0))
    traj_q_list = [agent.new_queue((Role.Server, 0), (Role.Learner, i)) for i in
                   range(agent.role_size(Role.Learner))]

    # syn net weights from learner
    all_weights = [param_q.get(dtype=weight.dtype, shape=weight.shape) for weight in rl_agent.all_weights]
    rl_agent.update_model(all_weights)

    train_cnt = 0
    action_dtype = tf.int32 if isinstance(rl_agent.actor.action_space, spaces.Discrete) else tf.float32

    curr_step = 0

    total_reward_list = []
    curr_reward_list = []
    tmp_eps_reward = 0
    while curr_step < total_step:
        # tmp_eps_reward = 0  # todo env with no end
        for _ in range(traj_len):
            curr_step += agent.role_size(Role.Actor)

            state_list = []
            for step_data_q in step_data_q_list:
                state_list.append(step_data_q.get(dtype=tf.float32, shape=obs_shape))

            action_list, log_p_list = rl_agent.get_action(state_list, batch_data=True)

            for action_q, action in zip(action_q_list, action_list):
                action_q.put(tf.Variable(action, dtype=action_dtype))
            reward_list, done_list, next_state_list = [], [], [],
            for i, step_data_q in enumerate(step_data_q_list):
                reward = step_data_q.get(dtype=tf.float32, shape=())
                if i == 0:
                    tmp_eps_reward += reward
                reward_list.append(reward)
                done = step_data_q.get(dtype=tf.float32, shape=())
                if i == 0 and done:
                    curr_reward_list.append(tmp_eps_reward)
                    tmp_eps_reward = 0
                done_list.append(done)
                next_state_list.append(step_data_q.get(dtype=tf.float32, shape=obs_shape))
            rl_agent.collect_data(state_list, action_list, reward_list, done_list, next_state_list, log_p_list, True)

        rl_agent.update_traj_list()

        # send traj to each learner and update weight
        learn_traj_len = learner_size * train_n_traj
        if len(rl_agent.traj_list) >= learn_traj_len:
            train_cnt += 1

            # todo env with end
            avg_eps_reward = None
            if curr_reward_list:
                avg_eps_reward = np.mean(curr_reward_list)
                curr_reward_list.clear()
                total_reward_list.append(avg_eps_reward)

            # todo env with no end
            # avg_eps_reward = tmp_eps_reward
            # total_reward_list.append(np.array(avg_eps_reward))

            print('Training iters: {}, steps so far: {}, average eps reward: {}'.format(
                train_cnt, curr_step, np.array(avg_eps_reward)))

            rl_agent.plot_save_log(total_reward_list, env_name)

            traj_iter = iter(rl_agent.traj_list[:learn_traj_len])
            rl_agent.traj_list = rl_agent.traj_list[learn_traj_len:]

            # send traj data to each learner
            for i, traj_q in enumerate(traj_q_list):
                for _ in range(train_n_traj):
                    try:
                        traj_data = next(traj_iter)
                    except StopIteration:
                        break
                    for data in traj_data:
                        traj_q.put(tf.Variable(data, dtype=tf.float32))

            # syn net weights from learner
            all_weights = [param_q.get(dtype=weight.dtype, shape=weight.shape) for weight in rl_agent.all_weights]
            rl_agent.update_model(all_weights)

            # save model
            if not train_cnt % save_interval:
                rl_agent.save_ckpt(env_name)

    # save the final model
    rl_agent.save_ckpt(env_name)
    print('Server Finished.')


def main(training_conf, env_conf, agent_conf):
    args = parse_args()
    agent = Agent(n_learners=args.l, n_actors=args.a, n_servers=args.s)

    print('%s : %d/%d' % (agent.role(), agent.role_rank(), agent.role_size()))

    agent.barrier()

    if agent.role() == Role.Learner:
        run_learner(agent, args, training_conf, env_conf, agent_conf)
    elif agent.role() == Role.Actor:
        run_actor(agent, args, training_conf, env_conf)
    elif agent.role() == Role.Server:
        run_server(agent, args, training_conf, env_conf, agent_conf)
    else:
        raise RuntimeError('Invalid Role.')

    agent.barrier()
