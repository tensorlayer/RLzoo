"""
Distributed Proximal Policy Optimization (DPPO)
----------------------------
A distributed version of OpenAI's Proximal Policy Optimization (PPO).
Workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

Reference
---------
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""

import argparse
import os
import queue
import threading
import time

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

from common.utils import *

EPS = 1e-8  # epsilon


class DPPO_PENALTY(object):
    """
    PPO class
    """

    def __init__(self, net_list, optimizers_list, state_dim, action_dim, a_bounds, kl_target=0.01, lam=0.5):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param state_dim: dimension of action for the environment
        :param action_dim: dimension of state for the environment
        :param a_bounds: a list of [min_action, max_action] action bounds for the environment
        :param kl_target: controls bounds of policy update and adaptive lambda
        :param lam:  KL-regularization coefficient
        """
        assert len(net_list) == 3
        assert len(optimizers_list) == 2
        self.name = 'dppo_penalty'
        self.kl_target = kl_target
        self.lam = lam
        if a_bounds[0] == a_bounds[1]:
            raise ValueError('a_bounds value error: min == max')
        self.a_bounds = a_bounds
        self.a_mean = np.mean(a_bounds, 0)
        self.a_scale = a_bounds[1] - self.a_mean

        self.critic, self.actor, self.actor_old = net_list

        self.critic_opt, self.actor_opt = optimizers_list

    def a_train(self, tfs, tfa, tfadv):
        """
        Update policy network
        :param tfs: state
        :param tfa: act
        :param tfadv: advantage
        :return:
        """
        tfs = np.array(tfs, np.float32)
        tfa = np.array(tfa, np.float32)
        tfadv = np.array(tfadv, np.float32)
        with tf.GradientTape() as tape:
            mu, log_sigma = self.actor(tfs)
            sigma = tf.math.exp(log_sigma)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, log_sigma_old = self.actor_old(tfs)
            sigma_old = tf.math.exp(log_sigma_old)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + EPS)
            surr = ratio * tfadv
            kl = tfp.distributions.kl_divergence(oldpi, pi)
            kl_mean = tf.reduce_mean(kl)
            aloss = -(tf.reduce_mean(surr - self.lam * kl))
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        # exit()
        return kl_mean

    def update_old_pi(self):
        """
        Update old policy parameter
        :return: None
        """
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    def c_train(self, tfdc_r, s):
        """
        Update actor network
        :param tfdc_r: cumulative reward
        :param s: state
        :return: None
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = tfdc_r - v
            closs = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        """
        Calculate advantage
        :param tfs: state
        :param tfdc_r: cumulative reward
        :return: advantage
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def update(self, a_update_steps, c_update_steps, save_interval):
        """
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        """
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.update_old_pi()  # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                s, a, r = zip(*data)
                s, a, r = np.vstack(s), np.vstack(a), np.vstack(r)
                s, a, r = np.array(s, np.float32), np.array(a, np.float32), np.array(r, np.float32)
                a = (a - self.a_mean) / self.a_scale

                adv = self.cal_adv(s, r)
                # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

                # update actor
                for _ in range(a_update_steps):
                    kl = self.a_train(s, a, adv)
                    if kl > 4 * self.kl_target:  # this in in google's paper
                        break
                if kl < self.kl_target / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    self.lam /= 2
                elif kl > self.kl_target * 1.5:
                    self.lam *= 2
                self.lam = np.clip(self.lam, 1e-4, 10)  # sometimes explode, this clipping is MorvanZhou's solution

                # update critic
                for _ in range(c_update_steps):
                    self.c_train(r, s)

                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

                if GLOBAL_EP and not GLOBAL_EP % save_interval:
                    self.save_ckpt()

    def get_action(self, s):
        """
        Choose action
        :param s: state
        :return: clipped act
        """
        s = s[np.newaxis, :].astype(np.float32)
        mu, log_sigma = self.actor(s)
        sigma = tf.math.exp(log_sigma)
        pi = tfp.distributions.Normal(mu, sigma)
        a = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        a = a * self.a_scale + self.a_mean
        a_out = np.clip(a, self.a_bounds[0], self.a_bounds[1])
        return a_out

    def get_v(self, s):
        """
        Compute value
        :param s: state
        :return: value
        """
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        save_model(self.actor, 'actor', 'ppo', )
        save_model(self.actor_old, 'actor_old', 'ppo', )
        save_model(self.critic, 'critic', 'ppo', )

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        load_model(self.actor, 'actor', self.name, )
        load_model(self.actor_old, 'actor_old', self.name, )
        load_model(self.critic, 'critic', self.name, )

    def learn(self, env, train_episodes=1000, test_episodes=100, max_steps=200, save_interval=10, gamma=0.9, seed=1,
              mode='train', batch_size=32, a_update_steps=10, c_update_steps=10, n_worker=4, reward_shaping=None):
        """
        learn function
        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps:  maximum number of steps for one episode
        :param save_interval: timesteps for saving
        :param gamma: reward discount factor
        :param seed: random seed
        :param mode: train or test
        :param batch_size: udpate batchsize
        :param a_update_steps: actor update iteration steps
        :param c_update_steps: critic update iteration steps
        :param n_worker: number of workers
        :param reward_shaping: reward shaping function
        :return: None
        """

        global GLOBAL_PPO, UPDATE_EVENT, ROLLING_EVENT, GLOBAL_UPDATE_COUNTER, GLOBAL_EP, GLOBAL_RUNNING_R, COORD, QUEUE
        global GAME, RANDOMSEED, EP_LEN, MIN_BATCH_SIZE, GAMMA, EP_MAX, REWARD_SHAPING
        GAME, RANDOMSEED, EP_LEN, MIN_BATCH_SIZE, GAMMA, EP_MAX = env, seed, max_steps, batch_size, gamma, train_episodes
        GLOBAL_PPO, REWARD_SHAPING = self, reward_shaping
        if mode == 'train':  # train
            UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
            UPDATE_EVENT.clear()  # not update now
            ROLLING_EVENT.set()  # start to roll out
            workers = [Worker(wid=i) for i in range(n_worker)]

            GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
            GLOBAL_RUNNING_R = []
            COORD = tf.train.Coordinator()
            QUEUE = queue.Queue()  # workers putting data in this queue
            threads = []
            for worker in workers:  # worker threads
                t = threading.Thread(target=worker.work, args=())
                t.start()  # training
                threads.append(t)
            # add a PPO updating thread
            threads.append(threading.Thread(target=self.update, args=(a_update_steps, c_update_steps, save_interval)))
            threads[-1].start()
            COORD.join(threads)

            self.save_ckpt()


        # test
        elif mode is 'test':
            self.load_ckpt()
            for _ in range(test_episodes):
                s = env.reset()
                for t in range(EP_LEN):
                    env.render()
                    s, r, done, info = env.step(self.get_action(s))
                    if done:
                        break
        else:
            print('unknown mode type')


class Worker(object):
    """
    Worker class for distributional running
    """

    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME.spec.id).unwrapped
        self.env.seed(wid * 100 + RANDOMSEED)
        self.ppo = GLOBAL_PPO

    def work(self):
        """
        Define a worker
        :return: None
        """
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, REWARD_SHAPING
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            t0 = time.time()
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.get_action(s)
                s_, r, done, _ = self.env.step(a)
                shaped_reward = REWARD_SHAPING(r) if REWARD_SHAPING else r  # normalize reward, find to be useful
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(shaped_reward)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put((bs, ba, br))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1

            print(
                'Episode: {}/{}  | Worker: {} | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    GLOBAL_EP, EP_MAX, self.wid, ep_r,
                    time.time() - t0
                )
            )
