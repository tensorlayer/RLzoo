from rlzoo.common.policy_networks import StochasticPolicyNetwork
from rlzoo.common.value_networks import ValueNetwork
import numpy as np
import copy
import pickle


def write_log(text: str):
    pass
    # print('infer server: '+text)
    # with open('infer_server_log.txt', 'a') as f:
    #     f.write(str(text) + '\n')


class DPPOInferServer:
    def __init__(self, net_builder, n_step=100, gamma=0.9):
        self.critic, self.actor = None, None
        self.net_builder = net_builder
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.logp_buffer = []
        self.gamma = gamma
        self.n_step = n_step

    def init_components(self):
        networks = self.net_builder()
        assert len(networks) == 2
        self.critic, self.actor = networks
        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

    def _cal_adv(self):
        dc_r = self._cal_discounted_r()
        s_shape = np.shape(self.state_buffer)
        s = np.reshape(self.state_buffer, [-1, s_shape[-1]])
        v = self.critic(s).numpy().reshape([-1, s_shape[1]])
        dc_r = np.array(dc_r, dtype=np.float32)
        advs = dc_r - v
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
        return advs

    def _get_v(self, s):
        return np.reshape(self.critic(s.astype(np.float32)), [-1])

    def _cal_discounted_r(self):
        discounted_r = np.zeros_like(self.reward_buffer)  # compute discounted reward
        v_s_ = self._get_v(self.state_buffer[-1]) * (1 - self.done_buffer[-1])
        for i in range(len(self.reward_buffer) - 1, -1, -1):
            discounted_r[i] = v_s_ = self.reward_buffer[i] + (1 - self.done_buffer[i]) * self.gamma * v_s_
        return discounted_r

    def _get_traj(self):
        traj_list = []
        for element in [self.state_buffer, self.action_buffer, self.reward_buffer, self.done_buffer, self._cal_adv(),
                        self.logp_buffer]:
            axes = list(range(len(np.shape(element))))
            axes[0], axes[1] = 1, 0
            traj_list.append(np.transpose(element, axes))
            if type(element) == list:
                element.clear()
        traj_list = list(zip(*traj_list))
        return traj_list

    def inference_service(self, batch_s):
        write_log('get action')
        # write_log(self.actor.trainable_weights)
        # write_log(batch_s)
        batch_s = np.array(batch_s)
        batch_a = self.actor(batch_s).numpy()
        write_log('get log p')
        batch_log_p = self.actor.policy_dist.get_param()
        return batch_a, batch_log_p

    def collect_data(self, s, a, r, d, log_p):
        self.state_buffer.append(s)
        self.action_buffer.append(a)
        self.reward_buffer.append(r)
        self.done_buffer.append(d)
        self.logp_buffer.append(log_p)

    def upload_data(self, que):
        traj_list = self._get_traj()
        traj = []
        for traj in traj_list:
            que.put(traj)
        # print('\rinfer server: updated, queue size: {}, current data shape: {}'.format(que.qsize(), [np.shape(i) for i in traj]))
        write_log('\rupdated, queue size: {}, current data shape: {}'.format(que.qsize(), [np.shape(i) for i in traj]))

    def run(self, pipe_list, traj_queue, should_stop, should_update, barrier, param_que):
        self.init_components()
        data = []
        for i, remote_connect in enumerate(pipe_list):
            write_log('recv {}'.format(i))
            data.append(remote_connect.recv())
        write_log('first recved')
        states, rewards, dones, infos = zip(*data)
        # states, rewards, dones, infos = zip(*[remote.recv() for remote in pipe_list])
        states, rewards, dones, infos = np.stack(states), np.stack(rewards), np.stack(dones), np.stack(infos)
        write_log('before while')
        while not should_stop.is_set():
            write_log('into while')
            if should_update.is_set():
                write_log('update_model')
                self.update_model(param_que)
                write_log('barrier.wait')
                barrier.wait()
            write_log('befor infer')
            actions, log_ps = self.inference_service(states)
            write_log('before send')
            for (remote, a) in zip(pipe_list, actions):
                remote.send(a)
            write_log('recv from pipe')
            states, rewards, dones, infos = zip(*[remote.recv() for remote in pipe_list])
            states, rewards, dones, infos = np.stack(states), np.stack(rewards), np.stack(dones), np.stack(infos)
            self.collect_data(states, actions, rewards, dones, log_ps)

            write_log('sampling, {}'.format(len(self.state_buffer)))
            # print('\rsampling, {}'.format(len(self.state_buffer)), end='')
            if len(self.state_buffer) >= self.n_step:
                self.upload_data(traj_queue)

    def update_model(self, param_que):
        write_log('get from param_que')
        params = param_que.get()
        write_log('assign param')
        for i, j in zip(self.critic.trainable_weights + self.actor.trainable_weights, params):
            i.assign(j)
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.logp_buffer.clear()

