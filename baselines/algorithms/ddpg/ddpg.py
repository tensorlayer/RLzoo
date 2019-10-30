"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""

import time

import numpy as np

import tensorflow as tf
import tensorlayer as tl
import gym

from common.utils import *
from common.buffer import *


###############################  DDPG  ####################################


class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, net_list, optimizers_list, replay_buffer_size, tau=0.01, var=3):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param replay_buffer_size: the size of buffer for storing explored samples
        :param tau: soft update factor
        :param var: control exploration
        """
        assert len(net_list) == 4
        assert len(optimizers_list) == 2
        self.name = 'ddpg'

        self.critic, self.critic_target, self.actor, self.actor_target = net_list

        def copy_para(from_model, to_model):
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        copy_para(self.actor, self.actor_target)
        copy_para(self.critic, self.critic_target)

        self.replay_buffer_size = replay_buffer_size
        self.buffer = ReplayBuffer(replay_buffer_size)

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement
        self.var = var

        self.critic_opt, self.actor_opt = optimizers_list

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: action
        """
        return self.actor([s])[0].numpy()

    def update(self, batch_size, gamma):
        """
        Update parameters
        :param batch_size: update batch size
        :param gamma: reward decay factor
        :return:
        """
        self.var *= 0.995
        bs, ba, br, bs_, bd = self.buffer.sample(batch_size)

        ba_ = self.actor_target(bs_)

        q_ = self.critic_target([bs_, ba_])
        y = br + (1 - bd) * gamma * q_
        with tf.GradientTape() as tape:
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = - tf.reduce_mean(q)  # maximize the q
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        self.ema_update()

    def store_transition(self, s, a, r, s_, d):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        d = 1 if d else 0

        self.buffer.push(s, a, [r], s_, d)

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        save_model(self.actor, 'actor', self.name, )
        save_model(self.actor_target, 'actor_target', self.name, )
        save_model(self.critic, 'critic', self.name, )
        save_model(self.critic_target, 'critic_target', self.name, )

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        load_model(self.actor, 'actor', 'ddpg', )
        load_model(self.actor_target, 'actor_target', 'ddpg', )
        load_model(self.critic, 'critic', 'ddpg', )
        load_model(self.critic_target, 'critic_target', 'ddpg', )

    def learn(self, env, train_episodes=200, test_episodes=100, max_steps=200, save_interval=10,
              mode='train', render=False, batch_size=32, gamma=0.9, seed=None):
        """
        learn function
        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: time steps for saving
        :param mode: train or test mode
        :param render: render each step
        :param batch_size: update batch size
        :param gamma: reward decay factor
        :param seed: random seed
        :return: None
        """
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        t0 = time.time()

        if mode == 'train':  # train
            reward_buffer = []
            for i in range(1, train_episodes + 1):
                s = env.reset()
                if render:
                    env.render()
                ep_reward = 0
                for j in range(max_steps):
                    # Add exploration noise
                    a = self.choose_action(s)
                    if isinstance(env.action_space, gym.spaces.Box):
                        a = np.clip(np.random.normal(a, self.var), env.action_space.low, env.action_space.high)
                    # add randomness to action selection for exploration

                    s_, r, done, info = env.step(a)

                    self.store_transition(s, a, r, s_, done)

                    if len(self.buffer) >= self.replay_buffer_size:
                        self.update(batch_size, gamma)
                    # print('s_', s_)
                    s = s_
                    ep_reward += r
                    if j == max_steps - 1:
                        print(
                            '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                i, train_episodes, ep_reward,
                                time.time() - t0
                            ), end=''
                        )
                    plt.show()
                # test
                if i and not i % save_interval:
                    s = env.reset()
                    ep_reward = 0
                    for j in range(max_steps):

                        a = self.choose_action(s)  # without exploration noise
                        s_, r, done, info = env.step(a)

                        s = s_
                        ep_reward += r
                        if j == max_steps - 1:
                            print(
                                '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                    i, train_episodes, ep_reward,
                                    time.time() - t0
                                )
                            )

                            reward_buffer.append(ep_reward)
                            self.save_ckpt()
            print('\nRunning time: ', time.time() - t0)

        # test
        elif mode == 'test':
            self.load_ckpt()
            ep_rs_sum = 0
            for eps in range(test_episodes):
                s = env.reset()
                for step in range(max_steps):
                    if render:
                        env.render()
                    action = self.choose_action(s)
                    observation, reward, done, info = env.step(action)
                    ep_rs_sum += reward
                    if done:
                        break
                try:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                except:
                    running_reward = ep_rs_sum

                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    eps, test_episodes, ep_rs_sum, time.time() - t0)
                )
        else:
            print('unknown mode type')
