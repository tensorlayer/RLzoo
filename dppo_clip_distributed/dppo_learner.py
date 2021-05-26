import queue

from rlzoo.common.utils import *
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *

EPS = 1e-8  # epsilon


def write_log(text: str):
    pass
    # print('learner: ' + text)
    # with open('learner_log.txt', 'a') as f:
    #     f.write(str(text) + '\n')


class DPPOLearner(object):
    """
    PPO class
    """

    def __init__(self, net_builder, epsilon=0.2):
        self.net_builder = net_builder
        self.name = 'DPPO_CLIP'
        self.epsilon = epsilon
        self.critic, self.actor = None, None

    def init_components(self):
        networks = self.net_builder()
        assert len(networks) == 2
        self.critic, self.actor = networks

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

    def update_model(self, param_que_list):
        params = param_que_list.get()
        for i, j in zip(self.critic.trainable_weights + self.actor.trainable_weights, params):
            i.assign(j)

    def run(self, traj_queue, grad_queue, should_stop, should_update, barrier, param_que_list,
            batch_length=10, a_update_steps=1, c_update_steps=1):  # todo a, c update step
        # todo max episode
        self.init_components()
        a_grad_queue, c_grad_queue = grad_queue
        batch_data = batch_s, batch_a, batch_r, batch_d, batch_adv, batch_logp = [], [], [], [], [], []

        while not should_stop.is_set():
            write_log('grad_queue size: {}'.format(grad_queue[0].qsize()))
            if should_update.is_set():
                write_log('update_model')
                self.update_model(param_que_list)
                write_log('barrier.wait')
                barrier.wait()
                for d in batch_data:
                    d.clear()
            write_log('get traj_queue {}'.format(traj_queue.qsize()))
            try:
                b_s, b_a, b_r, b_d, b_adv, b_logp = traj_queue.get(timeout=0.5)
                batch_s.append(b_s)
                batch_a.append(b_a)
                batch_r.append(b_r)
                batch_d.append(b_d)
                batch_adv.append(b_adv)
                batch_logp.append(b_logp)
            except queue.Empty:
                continue
            if len(batch_s) >= batch_length:
                write_log('batch data collected {}'.format([np.shape(i) for i in batch_data]))
                for s, a, r, d, adv, logp in zip(*batch_data):
                    s, a, r, d, adv, logp = np.vstack(s), np.vstack(a), np.vstack(r), np.vstack(d), \
                                            np.vstack(adv), np.vstack(logp)
                    s, a, r = np.array(s), np.array(a, np.float32), np.array(r, np.float32),
                    adv, logp = np.array(adv, np.float32), np.array(logp, np.float32),

                    # write_log('update actor')
                    # update actor
                    for _ in range(a_update_steps):
                        a_grad_queue.put(self.a_train(s, a, adv, logp))  # todo 这里待优化
                        # write_log('put a_grad_queue')

                    # write_log('update critic')
                    # update critic
                    for _ in range(c_update_steps):
                        c_grad_queue.put(self.c_train(r, s))  # todo 这里待优化
                        # write_log('put c_update_steps')

                for d in batch_data:
                    d.clear()

