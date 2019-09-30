"""
Vanilla Policy Gradient(VPG or REINFORCE)
-----------------------------------------
The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance.
It's an on-policy algorithm can be used for environments with either discrete or continuous action spaces.
Here is an example on discrete action space game CartPole-v0.
To apply it on continuous action space, you need to change the last softmax layer and the choose_action function.

Reference
---------
Cookbook: Barto A G, Sutton R S. Reinforcement Learning: An Introduction[J]. 1998.
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""
import time

import matplotlib.pyplot as plt
import numpy as np

import gym
import tensorflow as tf
import tensorlayer as tl

from common.utils import *
from common.buffer import *

###############################  PG  ####################################

class PolicyGradient:
    """
    PG class
    """
    def __init__(self, net_list, state_dim, action_dim, learning_rate=0.01, reward_decay=0.95):
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        [self.policy] = net_list 
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def choose_action(self, s):
        """
        choose action with probabilities.
        :param s: state
        :return: act
        """
        _logits = self.policy(np.array([s], np.float32))
        _probs = tf.nn.softmax(_logits).numpy()
        return tl.rein.choice_action_by_probs(_probs.ravel())

    def choose_action_greedy(self, s):
        """
        choose action with greedy policy
        :param s: state
        :return: act
        """
        _probs = tf.nn.softmax(self.policy(np.array([s], np.float32))).numpy()
        return np.argmax(_probs.ravel())

    def store_transition(self, s, a, r):
        """
        store data in memory buffer
        :param s: state
        :param a: act
        :param r: reward
        :return:
        """
        self.ep_obs.append(np.array([s], np.float32))
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def update(self):
        """
        update policy parameters via stochastic gradient ascent
        :return: None
        """
        # discount and normalize episode reward
        s, a, r = self.ep_obs, self.ep_as, self.ep_rs
        discounted_ep_rs_norm = self._discount_and_norm_rewards(r)

        with tf.GradientTape() as tape:
            # print(s)
            _logits = self.policy(np.vstack(s))
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=np.array(a))
            # this is negative log of chosen action

            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)

            loss = tf.reduce_mean(neg_log_prob * discounted_ep_rs_norm)  # reward guided loss

        grad = tape.gradient(loss, self.policy.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.policy.trainable_weights))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self, reward_list):
        """
        compute discount_and_norm_rewards
        :return: discount_and_norm_rewards
        """
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(reward_list)
        running_add = 0
        for t in reversed(range(0, len(reward_list))):
            running_add = running_add * self.gamma + reward_list[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save(self, name='model'):
        """
        save trained weights
        :return: None
        """
        save_model(self.policy, name, 'pg')

    def load(self, name='model'):
        """
        load trained weights
        :return: None
        """
        load_model(self.policy, name, 'pg')


    def learn(self, env, train_episodes=3000, test_episodes=1000, max_steps=1000, lr=0.02, gamma=0.99,
            seed=2, save_interval=100, mode='train', render=False):
        """
        learning parameters
        --------------------
        env: learning environment
        train_episodes: total number of episodes for training
        test_episodes: total number of episodes for testing
        max_steps: maximum number of steps for one episode
        lr: learning rate
        gamma: reward discount factor
        seed: random seed
        save_interval: timesteps for saving
        mode: train or test
        render: render each step for visualization
        """

        # reproducible
        np.random.seed(seed)
        tf.random.set_seed(seed)

        tl.logging.set_verbosity(tl.logging.DEBUG)
        if mode == 'train':
            reward_buffer = []
            for i_episode in range(train_episodes):

                episode_time = time.time()
                observation = env.reset()

                ep_rs_sum = 0
                for step in range(max_steps):
                    if render:
                        env.render()

                    action = self.choose_action(observation)

                    observation_, reward, done, info = env.step(action)

                    self.store_transition(observation, action, reward)

                    ep_rs_sum += reward
                    observation = observation_

                    if done:
                        break
                try:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                except:
                    running_reward = ep_rs_sum

                print(
                    "Episode [%d/%d] \tsum reward: %d  \trunning reward: %f \ttook: %.5fs " %
                    (i_episode, train_episodes, ep_rs_sum, running_reward, time.time() - episode_time)
                )
                reward_buffer.append(running_reward)

                self.update()

                if i_episode and i_episode % save_interval == 0:
                    self.save()
            plot(reward_buffer, 'pg', env.spec.id)
        
        elif mode == 'test':
            self.load()
            observation = env.reset()
            for eps in range(test_episodes):
                for step in range(max_steps):
                    env.render()
                    action = self.choose_action_greedy(observation)
                    observation, reward, done, info = env.step(action)
                    if done:
                        observation = env.reset()

        elif mode is not 'test':
            print('unknow mode type, activate test mode as default')
