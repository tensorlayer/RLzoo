"""
Deep Q Network
"""
import random
from copy import deepcopy

import numpy as np
import tensorflow as tf
from rlzoo.common.utils import *
from rlzoo.common.buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQN(object):
    """
    Papers:
    Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep
    reinforcement learning[J]. Nature, 2015, 518(7540): 529.
    Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining Improvements
    in Deep Reinforcement Learning[J]. 2017.
    """
    def __init__(self):
        self.name = 'DQN'

    def get_action(self, obv, out_dim, eps, qnet, mode):
        if mode == 'train' and random.random() < eps:
            return int(random.random() * out_dim)
        else:
            obv = np.expand_dims(obv, 0).astype('float32')
            return qnet(obv).numpy().argmax(1)[0]

    @staticmethod
    def sync(net, net_tar):
        """Copy q network to target q network"""
        for var, var_tar in zip(net.trainable_weights,
                                net_tar.trainable_weights):
            var_tar.assign(var)

    @tf.function
    def _td_error(self, transitions,
                  qnet, targetqnet, double, out_dim, reward_gamma):
        b_o, b_a, b_r, b_o_, b_d = transitions
        b_d = tf.cast(b_d, tf.float32)
        b_a = tf.cast(b_a, tf.int64)
        b_r = tf.cast(b_a, tf.float32)
        if double:
            b_a_ = tf.one_hot(tf.argmax(qnet(b_o_), 1), out_dim)
            b_q_ = (1 - b_d) * tf.reduce_sum(targetqnet(b_o_) * b_a_, 1)
        else:
            b_q_ = (1 - b_d) * tf.reduce_max(targetqnet(b_o_), 1)

        b_q = tf.reduce_sum(qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
        return b_q - (b_r + reward_gamma * b_q_)

    def learn(
            self, env,
            number_timesteps,
            network, optimizer,
            save_interval, test_episodes,
            gamma, exploration_rate, exploration_final_eps,
            double_q, target_network_update_freq, buffer_size,
            batch_size, train_freq, learning_starts,
            prioritized_replay, prioritized_alpha, prioritized_beta0,
            mode, checkpoint_path=None, save_path=None):
        """
        Parameters:
        ----------
        double_q (bool): if True double DQN will be used
        dueling (bool): if True dueling value estimation will be used
        exploration_rate (float): fraction of entire training period over
            which the exploration rate is annealed
        exploration_final_eps (float): final value of random action probability
        batch_size (int): size of a batched sampled from replay buffer for training
        train_freq (int): update the model every `train_freq` steps
        learning_starts (int): how many steps of the model to collect transitions
                               for before learning starts
        target_network_update_freq (int): update the target network every
                                          `target_network_update_freq` steps
        buffer_size (int): size of the replay buffer
        prioritized_replay (bool): if True prioritized replay buffer will be used.
        prioritized_alpha (float): alpha parameter for prioritized replay
        prioritized_beta0 (float): beta parameter for prioritized replay
        mode (str): train or test
        """
        if mode == 'train':
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            out_dim = env.action_space.n
            if prioritized_replay:
                buffer = PrioritizedReplayBuffer(
                    buffer_size,  prioritized_alpha, prioritized_beta0)
            else:
                buffer = ReplayBuffer(buffer_size)
            target_network = deepcopy(network)
            network.train()
            target_network.infer()
            parameters = network.trainable_weights

            o = env.reset()
            nepisode = 0
            for i in range(1, number_timesteps + 1):
                eps = 1 - (1 - exploration_final_eps) * \
                    min(1, i / exploration_rate * number_timesteps)
                a = self.get_action(o, out_dim, eps, network, mode)

                # execute action and feed to replay buffer
                # note that `_` tail in var name means next
                o_, r, done, info = env.step(a)
                buffer.push(o, a, r, o_, done)

                # update networks
                if i >= learning_starts and i % train_freq == 0:
                    if prioritized_replay:
                        # sample from prioritized replay buffer
                        *transitions, b_w, idxs = buffer.sample(batch_size)
                        # calculate weighted huber loss
                        with tf.GradientTape() as tape:
                            priorities = self._td_error(
                                transitions, network, target_network,
                                double_q, out_dim, gamma)
                            huber_loss = tf.where(tf.abs(priorities) < 1,
                                                tf.square(priorities) * 0.5,
                                                tf.abs(priorities) - 0.5)
                            loss = tf.reduce_mean(huber_loss * b_w)
                        # backpropagate
                        grad = tape.gradient(loss, parameters)
                        optimizer.apply_gradients(zip(grad, parameters))
                        # update priorities
                        priorities = np.clip(np.abs(priorities), 1e-6, None)
                        buffer.update_priorities(idxs, priorities)
                    else:
                        # sample from prioritized replay buffer
                        transitions = buffer.sample(batch_size)
                        # calculate huber loss
                        with tf.GradientTape() as tape:
                            td_errors = self._td_error(
                                transitions, network, target_network,
                                double_q, out_dim, gamma)
                            huber_loss = tf.where(tf.abs(td_errors) < 1,
                                                tf.square(td_errors) * 0.5,
                                                tf.abs(td_errors) - 0.5)
                            loss = tf.reduce_mean(huber_loss)
                        # backpropagate
                        grad = tape.gradient(loss, parameters)
                        optimizer.apply_gradients(zip(grad, parameters))
                if i % target_network_update_freq == 0:
                    self.sync(network, target_network)

                # reset current observation
                if done:
                    o = env.reset()
                else:
                    o = o_

                # episode in info is real (unwrapped) message
                if info.get('episode'):
                    nepisode += 1
                    reward, length = info['episode']['r'], info['episode']['l']
                    print(
                        'Time steps so far: {}, episode so far: {}, '
                        'episode reward: {:.4f}, episode length: {}'
                        .format(i, nepisode, reward, length)
                    )

                # saving model
                if i % save_interval == 0:
                    if save_path is not None:
                        network.save_weights(save_path)
                    else: # default
                        save_model(network, 'qnet', 'DQN', env.spec.id)


        elif mode == 'test':
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            out_dim = env.action_space.n
            if checkpoint_path is not None:
                network.load_weights(checkpoint_path)
            else:  # default
                load_model(network, 'qnet', 'DQN', env.spec.id)
            network.infer()
            nepisode = 0
            o = env.reset()
            while nepisode < test_episodes:
                a = self.get_action(o, out_dim, 0, network, mode)

                # execute action
                # note that `_` tail in var name means next
                o_, r, done, info = env.step(a)

                if done:
                    o = env.reset()
                else:
                    o = o_

                # episode in info is real (unwrapped) message
                if info.get('episode'):
                    nepisode += 1
                    reward, length = info['episode']['r'], info['episode']['l']
                    print(
                        'episode so far: {}, '
                        'episode reward: {:.4f}, episode length: {}'
                        .format(nepisode, reward, length)
                    )


        else:
            print('unknown mode type')
