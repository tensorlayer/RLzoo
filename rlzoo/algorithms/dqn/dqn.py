"""
Deep Q Network
"""
import random
from copy import deepcopy

from rlzoo.common.utils import *
from rlzoo.common.buffer import ReplayBuffer, PrioritizedReplayBuffer
from rlzoo.common.value_networks import *


class DQN(object):
    """
    Papers:

    Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep
    reinforcement learning[J]. Nature, 2015, 518(7540): 529.

    Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining Improvements
    in Deep Reinforcement Learning[J]. 2017.
    """

    def __init__(self, net_list, optimizers_list, double_q, dueling, buffer_size,
                 prioritized_replay, prioritized_alpha, prioritized_beta0, ):
        """
        Parameters:
        ----------
        :param net_list (list): a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list (list): a list of optimizers for all networks and differentiable variables
        :param double_q (bool): if True double DQN will be used
        :param dueling (bool): if True dueling value estimation will be used
        :param buffer_size (int): size of the replay buffer
        :param prioritized_replay (bool): if True prioritized replay buffer will be used.
        :param prioritized_alpha (float): alpha parameter for prioritized replay
        :param prioritized_beta0 (float): beta parameter for prioritized replay
        """
        assert isinstance(net_list[0], QNetwork)
        self.name = 'DQN'
        if prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size, prioritized_alpha, prioritized_beta0)
        else:
            self.buffer = ReplayBuffer(buffer_size)

        self.network = net_list[0]
        self.target_network = deepcopy(net_list[0])
        self.network.train()
        self.target_network.infer()
        self.optimizer = optimizers_list[0]
        self.double_q = double_q
        self.prioritized_replay = prioritized_replay
        self.dueling = dueling

    def get_action(self, obv, eps=0.2):
        out_dim = self.network.action_shape[0]
        if random.random() < eps:
            return int(random.random() * out_dim)
        else:
            obv = np.expand_dims(obv, 0).astype('float32')
            return self.network(obv).numpy().argmax(1)[0]

    def get_action_greedy(self, obv):
        obv = np.expand_dims(obv, 0).astype('float32')
        return self.network(obv).numpy().argmax(1)[0]

    def sync(self):
        """Copy q network to target q network"""

        for var, var_tar in zip(self.network.trainable_weights,
                                self.target_network.trainable_weights):
            var_tar.assign(var)

    def save_ckpt(self, env_name):
        """
        save trained weights
        :return: None
        """
        save_model(self.network, 'qnet', 'DQN', env_name)

    def load_ckpt(self, env_name):
        """
        load trained weights
        :return: None
        """
        load_model(self.network, 'qnet', 'DQN', env_name)

    # @tf.function
    def _td_error(self, transitions, reward_gamma):
        b_o, b_a, b_r, b_o_, b_d = transitions
        b_d = tf.cast(b_d, tf.float32)
        b_a = tf.cast(b_a, tf.int64)
        b_r = tf.cast(b_r, tf.float32)
        if self.double_q:
            b_a_ = tf.one_hot(tf.argmax(self.network(b_o_), 1), self.network.action_shape[0])
            b_q_ = (1 - b_d) * tf.reduce_sum(self.target_network(b_o_) * b_a_, 1)
        else:
            b_q_ = (1 - b_d) * tf.reduce_max(self.target_network(b_o_), 1)

        b_q = tf.reduce_sum(self.network(b_o) * tf.one_hot(b_a, self.network.action_shape[0]), 1)
        return b_q - (b_r + reward_gamma * b_q_)

    def store_transition(self, s, a, r, s_, d):
        self.buffer.push(s, a, r, s_, d)

    def update(self, batch_size, gamma):
        if self.prioritized_replay:
            # sample from prioritized replay buffer
            *transitions, b_w, idxs = self.buffer.sample(batch_size)
            # calculate weighted huber loss
            with tf.GradientTape() as tape:
                priorities = self._td_error(transitions, gamma)
                huber_loss = tf.where(tf.abs(priorities) < 1,
                                      tf.square(priorities) * 0.5,
                                      tf.abs(priorities) - 0.5)
                loss = tf.reduce_mean(huber_loss * b_w)
            # backpropagate
            grad = tape.gradient(loss, self.network.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.network.trainable_weights))
            # update priorities
            priorities = np.clip(np.abs(priorities), 1e-6, None)
            self.buffer.update_priorities(idxs, priorities)
        else:
            # sample from prioritized replay buffer
            transitions = self.buffer.sample(batch_size)
            # calculate huber loss
            with tf.GradientTape() as tape:
                td_errors = self._td_error(transitions, gamma)
                huber_loss = tf.where(tf.abs(td_errors) < 1,
                                      tf.square(td_errors) * 0.5,
                                      tf.abs(td_errors) - 0.5)
                loss = tf.reduce_mean(huber_loss)
            # backpropagate
            grad = tape.gradient(loss, self.network.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.network.trainable_weights))

    def learn(
            self, env, mode='train', render=False,
            train_episodes=1000, test_episodes=10, max_steps=200,
            save_interval=1000, gamma=0.99,
            exploration_rate=0.2, exploration_final_eps=0.01,
            target_network_update_freq=50,
            batch_size=32, train_freq=4, learning_starts=200,
            plot_func=None
    ):

        """
        :param env: learning environment
        :param mode: train or test
        :param render: render each step
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: time steps for saving
        :param gamma: reward decay factor
        :param exploration_rate (float): fraction of entire training period over
            which the exploration rate is annealed
        :param exploration_final_eps (float): final value of random action probability
        :param target_network_update_freq (int): update the target network every
                                          `target_network_update_freq` steps
        :param batch_size (int): size of a batched sampled from replay buffer for training
        :param train_freq (int): update the model every `train_freq` steps
        :param learning_starts (int): how many steps of the model to collect transitions
                               for before learning starts
        :param plot_func: additional function for interactive module

        """
        if mode == 'train':
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            i = 0
            for episode in range(1, train_episodes + 1):
                o = env.reset()
                ep_reward = 0
                for step in range(1, max_steps + 1):
                    i += 1
                    if render:
                        env.render()
                    eps = 1 - (1 - exploration_final_eps) * \
                          min(1, i / exploration_rate * (train_episodes * max_steps))
                    a = self.get_action(o, eps)

                    # execute action and feed to replay buffer
                    # note that `_` tail in var name means next
                    o_, r, done, info = env.step(a)
                    self.store_transition(o, a, r, o_, done)
                    ep_reward += r

                    # update networks
                    if i >= learning_starts and i % train_freq == 0:
                        self.update(batch_size, gamma)

                    if i % target_network_update_freq == 0:
                        self.sync()

                    # reset current observation
                    if done:
                        break
                    else:
                        o = o_

                    # saving model
                    if i % save_interval == 0:
                        self.save_ckpt(env.spec.id)
                print(
                    'Time steps so far: {}, episode so far: {}, '
                    'episode reward: {:.4f}, episode length: {}'
                        .format(i, episode, ep_reward, step)
                )
                reward_buffer.append(ep_reward)
                if plot_func is not None:
                    plot_func(reward_buffer)

        elif mode == 'test':
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))

            self.load_ckpt(env.spec.id)
            self.network.infer()

            reward_buffer = []
            for episode in range(1, test_episodes + 1):
                o = env.reset()
                ep_reward = 0
                for step in range(1, max_steps + 1):
                    if render:
                        env.render()
                    a = self.get_action_greedy(o)

                    # execute action
                    # note that `_` tail in var name means next
                    o_, r, done, info = env.step(a)
                    ep_reward += r

                    if done:
                        break
                    else:
                        o = o_

                print(
                    'episode so far: {}, '
                    'episode reward: {:.4f}, episode length: {}'
                        .format(episode, ep_reward, step)
                )
                reward_buffer.append(ep_reward)
                if plot_func is not None:
                    plot_func(reward_buffer)

        else:
            print('unknown mode type')
