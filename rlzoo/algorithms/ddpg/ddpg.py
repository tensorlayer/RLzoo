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
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
"""

import time

from rlzoo.common.utils import *
from rlzoo.common.buffer import *
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *


###############################  DDPG  ####################################


class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, net_list, optimizers_list, replay_buffer_size, action_range=1., tau=0.01):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param replay_buffer_size: the size of buffer for storing explored samples
        :param tau: soft update factor
        """
        assert len(net_list) == 4
        assert len(optimizers_list) == 2
        self.name = 'DDPG'

        self.critic, self.critic_target, self.actor, self.actor_target = net_list

        assert isinstance(self.critic, QNetwork)
        assert isinstance(self.critic_target, QNetwork)
        assert isinstance(self.actor, DeterministicPolicyNetwork)
        assert isinstance(self.actor_target, DeterministicPolicyNetwork)
        assert isinstance(self.actor.action_space, gym.spaces.Box)

        def copy_para(from_model, to_model):
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        copy_para(self.actor, self.actor_target)
        copy_para(self.critic, self.critic_target)

        self.replay_buffer_size = replay_buffer_size
        self.buffer = ReplayBuffer(replay_buffer_size)

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement
        self.action_range = action_range

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

    def sample_action(self):
        """ generate random actions for exploration """
        a = tf.random.uniform(self.actor.action_space.shape, self.actor.action_space.low, self.actor.action_space.high)
        return a

    def get_action(self, s, noise_scale):
        """
        Choose action with exploration

        :param s: state

        :return: action
        """
        a = self.actor([s])[0].numpy()*self.action_range

        # add randomness to action selection for exploration
        noise = np.random.normal(0, 1, a.shape) * noise_scale
        a += noise
        a = np.clip(a, self.actor.action_space.low, self.actor.action_space.high)

        return a

    def get_action_greedy(self, s):
        """
        Choose action

        :param s: state

        :return: action
        """
        return self.actor([s])[0].numpy()*self.action_range

    def update(self, batch_size, gamma):
        """
        Update parameters

        :param batch_size: update batch size
        :param gamma: reward decay factor

        :return:
        """
        bs, ba, br, bs_, bd = self.buffer.sample(batch_size)

        ba_ = self.actor_target(bs_)*self.action_range

        q_ = self.critic_target([bs_, ba_])
        y = br + (1 - bd) * gamma * q_
        with tf.GradientTape() as tape:
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(bs)*self.action_range
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

    def save_ckpt(self, env_name):
        """
        save trained weights

        :return: None
        """
        save_model(self.actor, 'model_policy_net', self.name, env_name)
        save_model(self.actor_target, 'model_target_policy_net', self.name, env_name)
        save_model(self.critic, 'model_q_net', self.name, env_name)
        save_model(self.critic_target, 'model_target_q_net', self.name, env_name)

    def load_ckpt(self, env_name):
        """
        load trained weights

        :return: None
        """
        load_model(self.actor, 'model_policy_net', self.name, env_name)
        load_model(self.actor_target, 'model_target_policy_net', self.name, env_name)
        load_model(self.critic, 'model_q_net', self.name, env_name)
        load_model(self.critic_target, 'model_target_q_net', self.name, env_name)

    def learn(self, env, train_episodes=200, test_episodes=100, max_steps=200, save_interval=10, explore_steps=500,
              mode='train', render=False, batch_size=32, gamma=0.9, noise_scale=1., noise_scale_decay=0.995,
              plot_func=None):
        """
        learn function

        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: time steps for saving
        :param explore_steps: for random action sampling in the beginning of training
        :param mode: train or test mode
        :param render: render each step
        :param batch_size: update batch size
        :param gamma: reward decay factor
        :param noise_scale: range of action noise for exploration
        :param noise_scale_decay: noise scale decay factor
        :param plot_func: additional function for interactive module
        :return: None
        """

        t0 = time.time()

        if mode == 'train':  # train
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            frame_idx = 0
            for i in range(1, train_episodes + 1):
                s = env.reset()
                ep_reward = 0

                for j in range(max_steps):
                    if render:
                        env.render()
                    # Add exploration noise
                    if frame_idx > explore_steps:
                        a = self.get_action(s, noise_scale)
                    else:
                        a = self.sample_action()
                        frame_idx += 1

                    s_, r, done, info = env.step(a)

                    self.store_transition(s, a, r, s_, done)
                    if len(self.buffer) >= self.replay_buffer_size:
                        self.update(batch_size, gamma)
                        noise_scale *= noise_scale_decay
                    s = s_
                    ep_reward += r

                    if done:
                        break

                print(
                    'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        i, train_episodes, ep_reward,
                        time.time() - t0
                    )
                )

                reward_buffer.append(ep_reward)
                if plot_func is not None:
                    plot_func(reward_buffer)
                if i and not i % save_interval:
                    self.save_ckpt(env_name=env.spec.id)
                    plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

            self.save_ckpt(env_name=env.spec.id)
            plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

        # test
        elif mode == 'test':
            self.load_ckpt(env_name=env.spec.id)
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            for eps in range(1, test_episodes+1):
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