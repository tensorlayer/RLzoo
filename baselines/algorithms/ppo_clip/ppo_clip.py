"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""
import time

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

from common.utils import *


EPS = 1e-8  # epsilon


###############################  PPO  ####################################

class PPO_CLIP(object):
    """
    PPO class
    """

    def __init__(self, net_list, optimizers_list, state_dim, action_dim, a_bounds, epsilon=0.2):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param state_dim: dimension of action for the environment
        :param action_dim: dimension of state for the environment
        :param a_bounds: a list of [min_action, max_action] action bounds for the environment
        :param epsilon: clip parameter
        """
        assert len(net_list) == 3
        assert len(optimizers_list) == 2
        if a_bounds[0] == a_bounds[1]:
            raise ValueError('a_bounds value error: min == max')
        self.a_bounds = a_bounds
        self.a_mean = np.mean(a_bounds, 0)
        self.a_scale = a_bounds[1] - self.a_mean

        self.epsilon = epsilon

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
            aloss = -tf.reduce_mean(
                tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * tfadv))
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

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

    def update(self, s, a, r, a_update_steps, c_update_steps):
        """
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        """
        s, a, r = s.astype(np.float32), a.astype(np.float32), r.astype(np.float32)
        a = (a - self.a_mean) / self.a_scale
        self.update_old_pi()
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # adv norm, sometimes helpful

        for _ in range(a_update_steps):
            self.a_train(s, a, adv)

        # update critic
        for _ in range(c_update_steps):
            self.c_train(r, s)

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
        save_model(self.actor, 'actor', 'ppo_clip', )
        save_model(self.actor_old, 'actor_old', 'ppo_clip', )
        save_model(self.critic, 'critic', 'ppo_clip', )

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        load_model(self.actor, 'actor', 'ppo_clip', )
        load_model(self.actor_old, 'actor_old', 'ppo_clip', )
        load_model(self.critic, 'critic', 'ppo_clip', )

    def learn(self, env, train_episodes=500, test_episodes=10, max_steps=200, save_interval=10,
              gamma=0.9, mode='train', render=False, batch_size=32, a_update_steps=10, c_update_steps=10, seed=1,
              reward_shaping=None):
        """
        learn function
        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: timesteps for saving
        :param gamma: reward discount factor
        :param mode: train or test
        :param render: render each step
        :param batch_size: udpate batchsize
        :param a_update_steps: actor update iteration steps
        :param c_update_steps: critic update iteration steps
        :param seed: random seed
        :param reward_shaping: reward shaping function
        :return: None
        """
        # reproducible
        env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        if mode == 'train':
            all_ep_r = []
            for ep in range(1, train_episodes + 1):
                s = env.reset()
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0
                t0 = time.time()
                for t in range(max_steps):  # in one episode
                    if render:
                        env.render()
                    a = self.get_action(s)
                    s_, r, done, _ = env.step(a)
                    shaped_reward = reward_shaping(r) if reward_shaping else r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(shaped_reward)
                    s = s_
                    ep_r += r

                    # update ppo
                    if (t + 1) % batch_size == 0 or t == max_steps - 1:
                        v_s_ = self.get_v(s_)
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                        buffer_s, buffer_a, buffer_r = [], [], []
                        self.update(bs, ba, br, a_update_steps, c_update_steps)
                if ep == 1:
                    all_ep_r.append(ep_r)
                else:
                    all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
                print(
                    'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        ep, train_episodes, ep_r,
                        time.time() - t0
                    )
                )

                if ep and not ep % save_interval:
                    self.save_ckpt()
            plot_save_log(all_ep_r, Algorithm_name='PPO_clip', Env_name=env.spec.id)

        # test
        elif mode is 'test':
            self.load_ckpt()
            for _ in range(test_episodes):
                s = env.reset()
                for i in range(max_steps):
                    env.render()
                    s, r, done, _ = env.step(self.get_action(s))
                    if done:
                        break
        else:
            print('unknown mode type')
