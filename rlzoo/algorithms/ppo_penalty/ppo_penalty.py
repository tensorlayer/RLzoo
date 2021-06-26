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
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""
import time

from rlzoo.common.utils import *
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *


EPS = 1e-8  # epsilon


class PPO_PENALTY(object):
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

        self.name = 'PPO_PENALTY'

        self.critic, self.actor = net_list

        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

        self.kl_target = kl_target
        self.lam = lam

        self.critic_opt, self.actor_opt = optimizers_list
        self.old_dist = make_dist(self.actor.action_space)

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
            kl = self.old_dist.kl(self.actor.policy_dist.param)
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

    def update(self, s, a, r, a_update_steps, c_update_steps):
        """
        Update parameter with the constraint of KL divergent

        :param s: state
        :param a: act
        :param r: reward

        :return: None
        """
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # normalize advantage, sometimes helpful

        _ = self.actor(s)
        oldpi_prob = tf.exp(self.actor.policy_dist.logp(a))
        oldpi_prob = tf.stop_gradient(oldpi_prob)

        oldpi_param = self.actor.policy_dist.get_param()
        self.old_dist.set_param(oldpi_param)

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

    def learn(self, env, train_episodes=1000, test_episodes=10, max_steps=200, save_interval=10, gamma=0.9,
              mode='train', render=False, batch_size=32, a_update_steps=10, c_update_steps=10,
              plot_func=None):
        """
        learn function

        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: time steps for saving
        :param gamma: reward discount factor
        :param mode: train or test
        :param render: render each step
        :param batch_size: update batch size
        :param a_update_steps: actor update iteration steps
        :param c_update_steps: critic update iteration steps
        :param plot_func: additional function for interactive module
        :return: None
        """

        t0 = time.time()
        if mode == 'train':
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            for ep in range(1, train_episodes + 1):
                s = env.reset()
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0
                for t in range(max_steps):  # in one episode
                    if render:
                        env.render()
                    a = self.get_action(s)
                    s_, r, done, _ = env.step(a)
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)  # normalize reward, find to be useful
                    s = s_
                    ep_r += r

                    # update ppo
                    if (t + 1) % batch_size == 0 or t == max_steps - 1:
                        if done:
                            v_s_ = 0
                        else:
                            try:
                                v_s_ = self.get_v(s_)
                            except:
                                v_s_ = self.get_v(s_[np.newaxis, :])   # for raw-pixel input
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bs = buffer_s
                        ba, br = np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                        buffer_s, buffer_a, buffer_r = [], [], []
                        self.update(bs, ba, br, a_update_steps, c_update_steps)

                    if done:
                        break

                print(
                    'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        ep, train_episodes, ep_r,
                        time.time() - t0
                    )
                )

                reward_buffer.append(ep_r)
                if plot_func is not None:
                    plot_func(reward_buffer)
                if ep and not ep % save_interval:
                    self.save_ckpt(env_name=env.spec.id)
                    plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

            self.save_ckpt(env_name=env.spec.id)
            plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

        # test
        elif mode == 'test':
            self.load_ckpt(env_name=env.spec.id)
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            for eps in range(test_episodes):
                ep_rs_sum = 0
                s = env.reset()
                for step in range(max_steps):
                    if render:
                        env.render()
                    action = self.get_action_greedy(s)
                    s, reward, done, info = env.step(action)
                    ep_rs_sum += reward
                    if done:
                        break

                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    eps, test_episodes, ep_rs_sum, time.time() - t0)
                )
            reward_buffer.append(ep_rs_sum)
            if plot_func:
                plot_func(reward_buffer)
        else:
            print('unknown mode type')

