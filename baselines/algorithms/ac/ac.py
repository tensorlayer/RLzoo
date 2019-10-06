"""
Actor-Critic 
-------------
It uses TD-error as the Advantage.

Actor Critic History
----------------------
A3C > DDPG > AC

Advantage
----------
AC converge faster than Policy Gradient.

Disadvantage (IMPORTANT)
------------------------
The Policy is oscillated (difficult to converge), DDPG can solve
this problem using advantage of DQN.

Reference
----------
paper: https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf
View more on MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
------------
CartPole-v0: https://gym.openai.com/envs/CartPole-v0

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The system is controlled by applying a force of +1 or -1
to the cart. The pendulum starts upright, and the goal is to prevent it from
falling over.

A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the
cart moves more than 2.4 units from the center.


Prerequisites
--------------
tensorflow >=2.0.0a0
tensorlayer >=2.0.0

"""
import argparse
import time

import numpy as np

import gym
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
from common.utils import *
from common.buffer import *
from common.value_networks import *
from common.policy_networks import *

tl.logging.set_verbosity(tl.logging.DEBUG)


###############################  Actor-Critic  ####################################
class AC():
    def __init__(self, net_list, optimizers_list, state_dim, action_dim, gamma=0.9):
        [self.actor, self.critic] = net_list
        [self.a_optimizer, self.c_optimizer] = optimizers_list
        self.GAMMA = gamma
        self.state_dim = state_dim

    def update(self, s, a, r, s_):
        s=s.astype(np.float32)
        s_=s_.astype(np.float32)
        # critic update
        v_ = self.critic(np.array([s_]))
        with tf.GradientTape() as tape:
            v = self.critic(np.array([s]))
            ## TD_error = r + lambd * V(newS) - V(S)
            td_error = r + self.GAMMA * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.c_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

        # actor update
        with tf.GradientTape() as tape:
            _logits = self.actor(np.array([s]))
            ## cross-entropy loss weighted by td-error (advantage),
            # the cross-entropy mearsures the difference of two probability distributions: the predicted logits and sampled action distribution,
            # then weighted by the td-error: small difference of real and predict actions for large td-error (advantage); and vice versa.
            _exp_v = tl.rein.cross_entropy_reward_loss(logits=_logits, actions=[a], rewards=td_error[0])
        grad = tape.gradient(_exp_v, self.actor.trainable_weights)
        self.a_optimizer.apply_gradients(zip(grad, self.actor.trainable_weights))
        return _exp_v

    def get_action(self, s):
        s=s.astype(np.float32)
        _logits = self.actor(np.array([s]))
        _probs = tf.nn.softmax(_logits).numpy()
        return tl.rein.choice_action_by_probs(_probs.ravel())  # sample according to probability distribution

    def choose_action_greedy(self, s):
        s=s.astype(np.float32)
        _logits = self.actor(np.array([s]))  # logits: probability distribution of actions
        _probs = tf.nn.softmax(_logits).numpy()
        return np.argmax(_probs.ravel())

    def save_ckpt(self):  # save trained weights
        save_model(self.actor, 'model_actor', 'AC')
        save_model(self.critic, 'model_critic', 'AC')

    def load_ckpt(self):  # load trained weights
        load_model(self.actor, 'model_actor', 'AC')
        load_model(self.critic, 'model_critic', 'AC')


    def learn(self, env, train_episodes, test_episodes=1000, max_steps=1000,
        seed=2, save_interval=100, mode='train', render=False):
        '''
        parameters
        -----------
        env: learning environment
        train_episodes:  total number of episodes for training
        test_episodes:  total number of episodes for testing
        max_steps:  maximum number of steps for one episode
        seed: random seed
        save_interval: timesteps for saving the weights and plotting the results
        mode: 'train' or 'test'
        render:  if true, visualize the environment
        '''

        env.seed(seed)  # reproducible
        np.random.seed(seed)
        tf.random.set_seed(seed)  # reproducible

        if mode=='train':
            t0 = time.time()
            rewards = []
            for i_episode in range(train_episodes):
                s = env.reset().astype(np.float32)
                t = 0  # number of step in this episode
                all_r = []  # rewards of all steps
                
                while True:

                    if render: env.render()

                    a = self.get_action(s)

                    s_new, r, done, info = env.step(a)
                    s_new = s_new

                    if done: r = -20

                    all_r.append(r)

                    try:
                        self.update(s, a, r, s_new)  # learn Policy : true_gradient = grad[logPi(s, a) * td_error]
                    except KeyboardInterrupt:  # if Ctrl+C at running actor.learn(), then save model, or exit if not at actor.learn()
                        self.save_ckpt()

                    s = s_new
                    t += 1

                    if done or t >= max_steps:
                        ep_rs_sum = sum(all_r)

                        if 'running_reward' not in globals():
                            running_reward = ep_rs_sum
                        else:
                            running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                        rewards.append(running_reward)

                        print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
                        .format(i_episode, train_episodes, ep_rs_sum, time.time()-t0 ))

                        # Early Stopping for quick check
                        if t >= max_steps:
                            print("Early Stopping")
                            s = env.reset().astype(np.float32)
                            rall = 0
                            while True:
                                env.render()
                                # a = actor.choose_action(s)
                                a = self.choose_action_greedy(s)  # Hao Dong: it is important for this task
                                s_new, r, done, info = env.step(a)
                                s_new = np.concatenate((s_new[0:self.state_dim], s[self.state_dim:]), axis=0).astype(np.float32)
                                rall += r
                                s = s_new
                                if done:
                                    s = env.reset().astype(np.float32)
                                    rall = 0
                        break

                    

                if i_episode%save_interval==0: 
                    self.save_ckpt()
                    plot(rewards, Algorithm_name='AC', Env_name=env.spec.id)
            self.save_ckpt()


        elif mode=='test':
            self.load_ckpt()
            t0 = time.time()

            for i_episode in range(test_episodes):
                episode_time = time.time()
                s = env.reset().astype(np.float32)
                t = 0  # number of step in this episode
                all_r = []  # rewards of all steps
                while True:
                    if render: env.render()
                    a = self.get_action(s)
                    s_new, r, done, info = env.step(a)
                    s_new = s_new.astype(np.float32)
                    if done: r = -20

                    all_r.append(r)
                    s = s_new
                    t += 1

                    if done or t >= max_steps:
                        ep_rs_sum = sum(all_r)

                        if 'running_reward' not in globals():
                            running_reward = ep_rs_sum
                        else:
                            running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                        print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
                        .format(i_episode, test_episodes, ep_rs_sum, time.time()-t0 ))

                        # Early Stopping for quick check
                        if t >= max_steps:
                            print("Early Stopping")
                            s = env.reset().astype(np.float32)
                            rall = 0
                            while True:
                                env.render()
                                # a = actor.choose_action(s)
                                a = self.choose_action_greedy(s)  # Hao Dong: it is important for this task
                                s_new, r, done, info = env.step(a)
                                s_new = np.concatenate((s_new[0:self.state_dim], s[self.state_dim:]), axis=0).astype(np.float32)
                                rall += r
                                s = s_new
                                if done:
                                    s = env.reset().astype(np.float32)
                                    rall = 0
                        break

        elif mode is not 'test':
            print('unknow mode type, activate test mode as default')
