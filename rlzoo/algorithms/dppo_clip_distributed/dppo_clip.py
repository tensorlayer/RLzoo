from rlzoo.common.policy_networks import StochasticPolicyNetwork
from rlzoo.common.value_networks import ValueNetwork
from rlzoo.common.utils import *
import tensorflow as tf
import numpy as np
import copy
import pickle


def write_log(text: str):
    pass
    # print('infer server: '+text)
    # with open('infer_server_log.txt', 'a') as f:
    #     f.write(str(text) + '\n')


EPS = 1e-8


class RLAlgorithm:
    def __init__(self):
        self.state_buffer = []  # shape: (None, [n_env], [state_shape])
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.next_state_buffer = []
        self.logp_buffer = []
        self.all_buffer = self.state_buffer, self.action_buffer, self.reward_buffer, self.done_buffer, \
                          self.next_state_buffer, self.logp_buffer
        self.traj_list = []
        self.gamma = 0.9
        self.name = 'NotNamed'

    @property
    def all_weights(self):
        raise NotImplementedError

    def update_model(self, params):
        raise NotImplementedError

    def _get_value(self, batch_state):
        """
        return: value: tf.Tensor
        """
        raise NotImplementedError

    def _get_action(self, batch_state):
        """
        return: action: tf.Tensor, log_p: tf.Tensor
        """
        raise NotImplementedError

    @property
    def logp_shape(self):
        raise NotImplementedError

    def save_ckpt(self, env_name):
        """
        save trained weights

        :return: None
        """
        raise NotImplementedError

    def plot_save_log(self, running_reward, env_name):
        plot_save_log(running_reward, algorithm_name=self.name, env_name=env_name)

    def collect_data(self, s, a, r, d, s_, log_p, batch_data=False):
        if not batch_data:
            s, a, r, d, s_, log_p = [s], [a], [r], [d], [s_], [log_p]
        for i, data in enumerate([s, a, r, d, s_, log_p]):
            self.all_buffer[i].append(data)

    def get_value(self, state, batch_data=False):
        if not batch_data:
            state = [state]
        value = self._get_value(np.array(state))
        value_shape = np.shape(value)
        value = tf.reshape(value, value_shape[:-1])
        return value

    def get_action(self, state, batch_data=False):
        if not batch_data:
            state = [state]

        state = np.array(state)
        action, log_p = self._get_action(state)
        action, log_p = action.numpy(), log_p.numpy()
        action_shape = np.shape(action)
        #      最后一维度是1                  是batch但是len=1就不转， 是batch本来要转
        #                                   不是batch时候len=1也要转
        if action_shape[-1] == 1 and batch_data ^ (len(action_shape) == 1):
            # ((batch_data and not len(action_shape) == 1) or (not batch_data and len(action_shape) == 1)):
            action = np.reshape(action, action_shape[:-1])  # 转换
            log_p = np.reshape(log_p, log_p.shape[:-1])
        return action, log_p

    # def _cal_discounted_r(self, state_list, reward_list, done_list, batch_data=False):
    #     discounted_r = []
    #     for r in reward_list[::-1]:
    #         v_s_ = r + 0.9 * v_s_
    #         discounted_r.append(v_s_)

    def _cal_discounted_r(self, next_state_list, reward_list, done_list, batch_data=False):
        discounted_r = np.zeros_like(reward_list)  # reward_buffer shape: [-1, n_env]
        # done_list = np.array(done_list, dtype=np.int)
        done_list = np.array(done_list)
        v_s_ = self.get_value(next_state_list[-1], batch_data) * (1 - done_list[-1])
        for i in range(len(reward_list) - 1, -1, -1):
            # discounted_r[i] = v_s_ = reward_list[i] + self.gamma * v_s_
            discounted_r[i] = v_s_ = reward_list[i] + (1 - done_list[i]) * self.gamma * v_s_
        return discounted_r

    def _cal_adv(self, state_list, reward_list, done_list, next_state_list, batch_data=False):
        dc_r = self._cal_discounted_r(next_state_list, reward_list, done_list, batch_data)
        # dc_r = np.array(
        #     [[6.5132155], [6.125795], [5.6953278], [5.217031], [4.68559], [4.0951], [3.439], [2.71], [1.9], [1.]])
        if batch_data:
            s_shape = np.shape(self.state_buffer)  # state_buffer shape: [-1, n_env, *obs_shape]
            state_list = np.reshape(self.state_buffer, [-1, *s_shape[2:]])
            v = self.get_value(state_list, batch_data).numpy()
            v = v.reshape(*s_shape[:2])
        else:
            v = self.get_value(state_list, batch_data).numpy()

        dc_r = np.array(dc_r, dtype=np.float32)
        advs = dc_r - v
        # advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)  # norm all env data adv at the same time
        return advs

    def _get_traj(self):
        traj_list = []
        for element in [
            self.state_buffer, self.action_buffer, self.reward_buffer, self.done_buffer, self.next_state_buffer,
            self._cal_adv(self.state_buffer, self.reward_buffer, self.done_buffer, self.next_state_buffer, True),
            self.logp_buffer]:
            axes = list(range(len(np.shape(element))))
            axes[0], axes[1] = 1, 0
            result = np.transpose(element, axes)
            # print(result)
            traj_list.append(result)
        traj_list = list(zip(*traj_list))  #
        return traj_list

    def update_traj_list(self):
        self.traj_list.extend(self._get_traj())
        for buffer in self.all_buffer:
            buffer.clear()


class DPPO_CLIP(RLAlgorithm):
    def __init__(self, net_builder, opt_builder, n_step=100, gamma=0.9, epsilon=0.2):
        super().__init__()
        self.critic, self.actor = None, None
        self.net_builder = net_builder
        self.gamma = gamma
        self.n_step = n_step
        self._logp_shape = None
        self.epsilon = epsilon
        self.name = 'DPPO_CLIP'
        self.acter_optimizer, self.critic_optimizer = opt_builder()

    def init_components(self):  # todo init process should be placed
        networks = self.net_builder()
        assert len(networks) == 2
        self.critic, self.actor = networks
        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

    @property
    def all_weights(self):
        return self.critic.trainable_weights + self.actor.trainable_weights

    # api
    def _get_action(self, state):
        action = self.actor(state)
        log_p = self.actor.policy_dist.logp(action)
        return action, log_p

    def _get_value(self, state):
        return self.critic(state)

    def save_ckpt(self, env_name):
        """
        save trained weights

        :return: None
        """
        save_model(self.actor, 'actor', self.name, env_name)
        save_model(self.critic, 'critic', self.name, env_name)

    def load_ckpt(self, env_name):
        """
        load trained weights

        :return: None
        """
        load_model(self.actor, 'actor', self.name, env_name)
        load_model(self.critic, 'critic', self.name, env_name)

    # api
    def update_model(self, params):
        for i, j in zip(self.all_weights, params):
            i.assign(j)
        for buffer in self.all_buffer:
            buffer.clear()

    def a_train(self, s, a, adv, oldpi_logp):
        oldpi_prob = tf.exp(oldpi_logp)
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
        dc_r = np.array(dc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = dc_r - v
            closs = tf.reduce_mean(tf.square(advantage))
        c_grad = tape.gradient(closs, self.critic.trainable_weights)
        return c_grad

    def train(self, traj_list, dis_agent=None):
        for traj in traj_list:
            state_list, action_list, reward_list, done_list, next_state_list, adv_list, logp_list = traj
            for _ in range(10):
                a_grad = self.a_train(state_list, action_list, adv_list, logp_list)
                if dis_agent:
                    a_grad = [dis_agent.role_all_reduce(grad) for grad in a_grad]
                self.acter_optimizer.apply_gradients(zip(a_grad, self.actor.trainable_weights))

            dc_r = self._cal_discounted_r(next_state_list, reward_list, done_list)
            for _ in range(10):
                c_grad = self.c_train(dc_r, state_list)
                if dis_agent:
                    c_grad = [dis_agent.role_all_reduce(grad) for grad in c_grad]
                self.critic_optimizer.apply_gradients(zip(c_grad, self.critic.trainable_weights))


if __name__ == '__main__':
    from rlzoo.distributed.training_components import net_builder, env_maker, opt_builder
    from rlzoo.common.utils import set_seed

    env = env_maker()
    # set_seed(1, env)

    agent = DPPO_CLIP(net_builder, opt_builder)
    agent.init_components()

    running_reward = []
    curr_step, max_step, traj_len = 0, 500 * 200, 200
    s = env.reset()
    d = False
    cnt = 0
    while curr_step < max_step:
        for _ in range(traj_len):
            curr_step += 1
            a, logp = agent.get_action(s)
            s_, r, d, _ = env.step(a)
            agent.collect_data(s, a, r, d, s_, logp)
            if d:
                s = env.reset()
            else:
                s = s_
        agent.update_traj_list()
        agent.train(agent.traj_list)
        avg_eps_reward = min(sum(agent.traj_list[0][2]) / (sum(agent.traj_list[0][3] + 1e-10)), traj_len)
        agent.traj_list.clear()
        running_reward.append(avg_eps_reward)
        cnt += 1
        print(cnt, curr_step, avg_eps_reward)
        agent.plot_save_log(running_reward, env.spec.id)
