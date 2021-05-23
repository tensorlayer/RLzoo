from rlzoo.common.policy_networks import StochasticPolicyNetwork
from rlzoo.common.value_networks import ValueNetwork
import numpy as np
import copy
import pickle


class DPPOInferServer:
    def __init__(self, net_builder, net_param_pipe, n_step=1000, gamma=0.9):
        networks = net_builder()
        assert len(networks) == 2
        self.critic, self.actor = networks
        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.logp_buffer = []
        self.gamma = gamma
        self.n_step = n_step
        self.net_param_pipe = net_param_pipe

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
        traj = []
        for element in [self.state_buffer, self.action_buffer, self.reward_buffer, self.done_buffer, self._cal_adv(),
                        self.logp_buffer]:
            axes = list(range(len(np.shape(element))))
            axes[0], axes[1] = 1, 0
            traj.append(np.transpose(element, axes))
            if type(element) == list:
                element.clear()
        return traj

    def inference_service(self, batch_s):
        print(batch_s)
        batch_a = self.actor(batch_s).numpy()
        batch_log_p = self.actor.policy_dist.get_param()
        return batch_a, batch_log_p

    def collect_data(self, s, a, r, d, log_p):
        self.state_buffer.append(s)
        self.action_buffer.append(a)
        self.reward_buffer.append(r)
        self.done_buffer.append(d)
        self.logp_buffer.append(log_p)

    def upload_data(self, que):
        traj_data = self._get_traj()
        que.put(traj_data)
        print('\rupdated, queue size: {}, current data shape: {}'.format(que.qsize(), [np.shape(i) for i in traj_data]))

    def run(self, pipe_list, traj_queue, should_stop, should_update, barrier, ):
        states, rewards, dones, infos = zip(*[remote.recv() for remote in pipe_list])
        states, rewards, dones, infos = np.stack(states), np.stack(rewards), np.stack(dones), np.stack(infos)

        while not should_stop.is_set():
            if should_update.is_set():
                self.update_model()
                barrier.wait()
            actions, log_ps = self.inference_service(states)
            for (remote, a) in zip(pipe_list, actions):
                remote.send(a)

            states, rewards, dones, infos = zip(*[remote.recv() for remote in pipe_list])
            states, rewards, dones, infos = np.stack(states), np.stack(rewards), np.stack(dones), np.stack(infos)
            self.collect_data(states, actions, rewards, dones, log_ps)

            print('\rsampling, {}'.format(len(self.state_buffer)), end='')
            if len(self.state_buffer) >= self.n_step:
                self.upload_data(traj_queue)

    def update_model(self):
        params = self.net_param_pipe.recv()
        for i, j in zip(self.critic.trainable_weights + self.actor.trainable_weights, params):
            i.assign(j)
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.logp_buffer.clear()


if __name__ == '__main__':
    import multiprocessing as mp

    from rlzoo.common.env_wrappers import build_env
    from dppo_clip_distributed.dppo_sampler import DPPOSampler
    import copy, json, pickle
    from gym.spaces.box import Box
    from gym.spaces.discrete import Discrete
    import cloudpickle

    should_stop_event = mp.Event()
    should_stop_event.clear()

    # build_sampler
    nenv = 3


    def build_func():
        return build_env('CartPole-v0', 'classic_control')


    pipe_list = []
    for _ in range(nenv):
        sampler = DPPOSampler(build_func)
        remote_a, remote_b = mp.Pipe()
        p = mp.Process(target=sampler.run, args=(remote_a, should_stop_event))
        p.daemon = True  # todo 守护进程的依赖关系
        p.start()
        pipe_list.append(remote_b)

    traj_queue = mp.Queue(maxsize=10000)
    grad_queue = mp.Queue(maxsize=10000), mp.Queue(maxsize=10000),
    should_update_event = mp.Event()
    should_update_event.clear()
    barrier = mp.Barrier(1)  # sampler + updater

    """ build networks for the algorithm """
    name = 'DPPO_CLIP'
    hidden_dim = 64
    num_hidden_layer = 2
    critic = ValueNetwork(Box(0, 1, (4,)), [hidden_dim] * num_hidden_layer, name=name + '_value')
    actor = StochasticPolicyNetwork(Box(0, 1, (4,)), Discrete(2),
                                    [hidden_dim] * num_hidden_layer,
                                    trainable=True,
                                    name=name + '_policy')

    actor = copy.deepcopy(actor)
    global_nets = critic, actor

    global_nets = cloudpickle.dumps(global_nets)
    # p = mp.Process(
    #     target=DPPOInferServer(global_nets).run,
    #     args=(traj_queue, should_stop_event, should_update_event, barrier)
    # )
    # p.start()
    DPPOInferServer(global_nets).run(pipe_list, traj_queue, should_stop_event, should_update_event, barrier)
