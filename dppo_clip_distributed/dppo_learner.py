import queue

from rlzoo.common.utils import *
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *

EPS = 1e-8  # epsilon


class DPPOLearner(object):
    """
    PPO class
    """

    def __init__(self, net_builder, net_param_pipe, epsilon=0.2):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param state_dim: dimension of action for the environment
        :param action_dim: dimension of state for the environment
        :param a_bounds: a list of [min_action, max_action] action bounds for the environment
        :param epsilon: clip parameter
        """
        networks = net_builder()
        assert len(networks) == 2
        self.name = 'DPPO_CLIP'

        self.epsilon = epsilon

        self.critic, self.actor = networks
        self.net_param_pipe = net_param_pipe

        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

    def a_train(self, s, a, adv, oldpi_prob):
        """
        Update policy network

        :param s: state
        :param a: act
        :param adv: advantage
        :param oldpi_prob: old pi probability of a in s

        :return:
        """
        with tf.GradientTape() as tape:
            _ = self.actor(s)
            pi_prob = tf.exp(self.actor.policy_dist.logp(a))
            ratio = pi_prob / (oldpi_prob + EPS)

            surr = ratio * adv
            aloss = -tf.reduce_mean(
                tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv))
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        return a_gard

    def c_train(self, dc_r, s):
        """
        Update actor network

        :param dc_r: cumulative reward
        :param s: state

        :return: None
        """
        dc_r = np.array(dc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = dc_r - v
            closs = tf.reduce_mean(tf.square(advantage))
        c_grad = tape.gradient(closs, self.critic.trainable_weights)
        return c_grad

    def update_model(self):
        params = self.net_param_pipe.recv()
        for i, j in zip(self.critic.trainable_weights + self.actor.trainable_weights, params):
            i.assign(j)

    def run(self, traj_queue, grad_queue, should_stop, should_update, barrier,
            batch_length=10, a_update_steps=1, c_update_steps=1):  # todo a, c update step
        # todo max episode
        a_grad_queue, c_grad_queue = grad_queue

        while not should_stop.is_set():
            if should_update.is_set():
                self.update_model()
                barrier.wait()
            batch_data = batch_s, batch_a, batch_r, batch_d, batch_adv, batch_logp = [], [], [], [], [], []
            for _ in range(batch_length):
                b_s, b_a, b_r, b_d, b_adv, b_logp = traj_queue.get()
                batch_s.extend(b_s)
                batch_a.extend(b_a)
                batch_r.extend(b_r)
                batch_d.extend(b_d)
                batch_adv.extend(b_adv)
                batch_logp.extend(b_logp)
            pass
            for s, a, r, d, adv, logp in zip(*batch_data):
                s, a, r, d, adv, logp = np.vstack(s), np.vstack(a), np.vstack(r), np.vstack(d), \
                                        np.vstack(adv), np.vstack(logp)
                s, a, r = np.array(s), np.array(a, np.float32), np.array(r, np.float32),
                adv, logp = np.array(adv, np.float32), np.array(logp, np.float32),

                # update actor
                for _ in range(a_update_steps):
                    a_grad_queue.put(self.a_train(s, a, adv, logp))  # todo 这里待优化

                # update critic
                for _ in range(c_update_steps):
                    c_grad_queue.put(self.c_train(r, s))  # todo 这里待优化


if __name__ == '__main__':
    import multiprocessing as mp
    import cloudpickle
    from rlzoo.common.env_wrappers import build_env
    import copy, json, pickle
    from gym.spaces.box import Box
    from gym.spaces.discrete import Discrete

    traj_queue = mp.Queue(maxsize=10000)
    grad_queue = mp.Queue(maxsize=10000), queue.Queue(maxsize=10000),
    should_stop_event = mp.Event()
    should_stop_event.clear()
    should_update_event = mp.Event()
    should_update_event.clear()
    barrier = mp.Barrier(2)  # sampler + updater

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

    with open('queue_data.json', 'rb') as file:
        queue_data = pickle.load(file)
    for data in queue_data:
        traj_queue.put(data)
    print(traj_queue.qsize())
    actor_lr = 1e-4
    critic_lr = 2e-4
    optimizers_list = [tf.optimizers.Adam(critic_lr), tf.optimizers.Adam(actor_lr)]
    dcu = DPPOLearner(global_nets, traj_queue, grad_queue, should_stop_event, should_update_event, barrier)
    global_nets = cloudpickle.dumps(global_nets)
    p = mp.Process(target=dcu.run, args=())
    p.daemon = True
    p.start()
