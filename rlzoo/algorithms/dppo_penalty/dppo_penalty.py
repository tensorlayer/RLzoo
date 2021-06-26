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
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""

import queue
import threading
import time

from rlzoo.common.utils import *
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *

EPS = 1e-8  # epsilon


class DPPO_PENALTY(object):
    """
    PPO class
    """

    def __init__(self, net_list, optimizers_list, kl_target=0.01, lam=0.5):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param kl_target: controls bounds of policy update and adaptive lambda
        :param lam:  KL-regularization coefficient
        """
        assert len(net_list) == 2
        assert len(optimizers_list) == 2
        self.name = 'DPPO_PENALTY'
        self.kl_target = kl_target
        self.lam = lam

        self.critic, self.actor = net_list

        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

        self.critic_opt, self.actor_opt = optimizers_list
        self.old_dist = make_dist(self.actor.action_space)
        self.last_update_epoch = 0

    def a_train(self, tfs, tfa, tfadv, oldpi_prob):
        """
        Update policy network
        
        :param tfs: state
        :param tfa: act
        :param tfadv: advantage

        :return:
        """
        tfs = np.array(tfs)
        tfa = np.array(tfa, np.float32)
        tfadv = np.array(tfadv, np.float32)
        with tf.GradientTape() as tape:
            _ = self.actor(tfs)
            pi_prob = tf.exp(self.actor.policy_dist.logp(tfa))
            ratio = pi_prob / (oldpi_prob + EPS)

            surr = ratio * tfadv
            kl = self.old_dist.kl(self.actor.policy_dist.get_param())
            # kl = tfp.distributions.kl_divergence(oldpi, pi)
            kl_mean = tf.reduce_mean(kl)
            aloss = -(tf.reduce_mean(surr - self.lam * kl))
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        return kl_mean

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

    def update(self, a_update_steps, c_update_steps, save_interval, env):
        """
        Update

        :param a_update_steps: actor update steps
        :param c_update_steps: critic update steps
        :param save_interval: save interval
        :param env: environment

        :return: None
        """
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                s, a, r = zip(*data)
                s, a, r = np.vstack(s), np.vstack(a), np.vstack(r)
                s, a, r = np.array(s), np.array(a, np.float32), np.array(r, np.float32)

                adv = self.cal_adv(s, r)
                # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

                _ = self.actor(s)
                oldpi_prob = tf.exp(self.actor.policy_dist.logp(a))
                oldpi_prob = tf.stop_gradient(oldpi_prob)

                oldpi_param = self.actor.policy_dist.get_param()
                self.old_dist.set_param(oldpi_param)

                # update actor
                for _ in range(a_update_steps):
                    kl = self.a_train(s, a, adv, oldpi_prob)
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

                if (not GLOBAL_EP % save_interval) and GLOBAL_EP != self.last_update_epoch:
                    self.save_ckpt(env_name=env.spec.id)
                    self.last_update_epoch = GLOBAL_EP
                    plot_save_log(GLOBAL_RUNNING_R, algorithm_name=self.name, env_name=env.spec.id)

    def get_action(self, s):
        """
        Choose action

        :param s: state

        :return: clipped act
        """
        return self.actor([s])[0].numpy()

    def get_action_greedy(self, s):
        """
        Choose action

        :param s: state

        :return: clipped act
        """
        return self.actor([s], greedy=True)[0].numpy()

    def get_v(self, s):
        """
        Compute value

        :param s: state

        :return: value
        """
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]

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

    def learn(self, env, train_episodes=200, test_episodes=100, max_steps=200, save_interval=10, gamma=0.9,
              mode='train', render=False, batch_size=32, a_update_steps=10, c_update_steps=10, n_workers=4,
              plot_func=None):
        """
        learn function

        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps:  maximum number of steps for one episode
        :param save_interval: time steps for saving
        :param gamma: reward discount factor
        :param mode: train or test
        :param render: render each step
        :param batch_size: update batch size
        :param a_update_steps: actor update iteration steps
        :param c_update_steps: critic update iteration steps
        :param n_workers: number of workers
        :param plot_func: additional function for interactive module
        :return: None
        """
        t0 = time.time()
        global GLOBAL_PPO, UPDATE_EVENT, ROLLING_EVENT, GLOBAL_UPDATE_COUNTER, GLOBAL_EP, GLOBAL_RUNNING_R, COORD, QUEUE
        global EP_LEN, MIN_BATCH_SIZE, GAMMA, EP_MAX, RENDER
        EP_LEN, MIN_BATCH_SIZE, GAMMA, EP_MAX, RENDER = max_steps, batch_size, gamma, train_episodes, render
        GLOBAL_PPO = self
        if mode == 'train':  # train
            if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
                assert len(env) == n_workers
            else:
                assert n_workers == 1
                env = env,
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env[0].spec.id))
            UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
            UPDATE_EVENT.clear()  # not update now
            ROLLING_EVENT.set()  # start to roll out
            workers = [Worker(wid=i, env=env[i], plot_func=plot_func) for i in range(n_workers)]

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
            threads.append(threading.Thread(target=self.update, args=(a_update_steps, c_update_steps,
                                                                      save_interval, env[0])))
            threads[-1].start()
            COORD.join(threads)

            self.save_ckpt(env_name=env[0].spec.id)
            plot_save_log(GLOBAL_RUNNING_R, algorithm_name=self.name, env_name=env[0].spec.id)

        # test
        elif mode == 'test':
            if isinstance(env, list):  # judge if multiple envs are passed in for parallel computing
                env = env[0]
            else:
                env = env
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            self.load_ckpt(env_name=env.spec.id)
            for eps in range(test_episodes):
                ep_rs_sum = 0
                s = env.reset()
                for step in range(max_steps):
                    if RENDER:
                        env.render()
                    action = self.get_action_greedy(s)
                    s, reward, done, info = env.step(action)
                    ep_rs_sum += reward
                    if done:
                        break

                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    eps, test_episodes, ep_rs_sum, time.time() - t0)
                )
        else:
            print('unknown mode type')


class Worker(object):
    """
    Worker class for distributional running
    """

    def __init__(self, wid, env, plot_func):
        self.wid = wid
        self.env = env
        global GLOBAL_PPO
        self.ppo = GLOBAL_PPO
        self.plot_func = plot_func

    def work(self):
        """
        Define a worker

        :return: None
        """
        t0 = time.time()
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, REWARD_SHAPING, RENDER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(1, EP_LEN + 1):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.get_action(s)
                for step in range(EP_LEN):
                    if RENDER:
                        self.env.render()
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    if done:
                        v_s_ = 0
                    else:
                        try:
                            v_s_ = self.ppo.get_v(s_)
                        except:
                            v_s_ = self.ppo.get_v(s_[np.newaxis, :])  # for raw-pixel input
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs = buffer_s
                    ba, br = np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put((bs, ba, br))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break
                if done:
                    break

            GLOBAL_RUNNING_R.append(ep_r)
            GLOBAL_EP += 1

            print(
                'Episode: {}/{}  | Worker: {} | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    GLOBAL_EP, EP_MAX, self.wid, ep_r,
                    time.time() - t0
                )
            )
            if self.wid == 0 and self.plot_func is not None:
                self.plot_func(GLOBAL_RUNNING_R)
