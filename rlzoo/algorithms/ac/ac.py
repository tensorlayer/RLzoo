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
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/

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
import time

import tensorlayer as tl

from rlzoo.common.utils import *
from rlzoo.common.value_networks import *
from rlzoo.common.policy_networks import *

tl.logging.set_verbosity(tl.logging.DEBUG)


###############################  Actor-Critic  ####################################
class AC:
    def __init__(self, net_list, optimizers_list, gamma=0.9):
        assert len(net_list) == 2
        assert len(optimizers_list) == 2
        self.name = 'AC'
        self.actor, self.critic = net_list
        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)
        self.a_optimizer, self.c_optimizer = optimizers_list
        self.GAMMA = gamma

    def update(self, s, a, r, s_):
        # critic update
        v_ = self.critic(np.array([s_]))
        with tf.GradientTape() as tape:
            v = self.critic(np.array([s]))
            td_error = r + self.GAMMA * v_ - v  # TD_error = r + lambd * V(newS) - V(S)
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.c_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

        # actor update
        with tf.GradientTape() as tape:
            # _logits = self.actor(np.array([s]))
            ## cross-entropy loss weighted by td-error (advantage),
            # the cross-entropy mearsures the difference of two probability distributions: the predicted logits and sampled action distribution,
            # then weighted by the td-error: small difference of real and predict actions for large td-error (advantage); and vice versa.

            _ = self.actor(np.array([s]))
            neg_log_prob = self.actor.policy_dist.neglogp([a])
            _exp_v = tf.reduce_mean(neg_log_prob * td_error)
        grad = tape.gradient(_exp_v, self.actor.trainable_weights)
        self.a_optimizer.apply_gradients(zip(grad, self.actor.trainable_weights))
        return _exp_v

    def get_action(self, s):
        return self.actor(np.array([s]))[0].numpy()

    def get_action_greedy(self, s):
        return self.actor(np.array([s]), greedy=True)[0].numpy()

    def save_ckpt(self, env_name):  # save trained weights
        save_model(self.actor, 'model_actor', self.name, env_name)
        save_model(self.critic, 'model_critic', self.name, env_name)

    def load_ckpt(self, env_name):  # load trained weights
        load_model(self.actor, 'model_actor', self.name, env_name)
        load_model(self.critic, 'model_critic', self.name, env_name)

    def learn(self, env, train_episodes=1000, test_episodes=500, max_steps=200,
              save_interval=100, mode='train', render=False, plot_func=None):
        """
        :param env: learning environment
        :param train_episodes:  total number of episodes for training
        :param test_episodes:  total number of episodes for testing
        :param max_steps:  maximum number of steps for one episode
        :param save_interval: time steps for saving the weights and plotting the results
        :param mode: 'train' or 'test'
        :param render:  if true, visualize the environment
        :param plot_func: additional function for interactive module
        """

        t0 = time.time()
        if mode == 'train':
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            for i_episode in range(train_episodes):
                s = env.reset()
                ep_rs_sum = 0  # rewards of all steps

                for step in range(max_steps):

                    if render:
                        env.render()

                    a = self.get_action(s)
                    s_new, r, done, info = env.step(a)
                    ep_rs_sum += r

                    try:
                        self.update(s, a, r, s_new)  # learn Policy : true_gradient = grad[logPi(s, a) * td_error]
                    except KeyboardInterrupt:  # if Ctrl+C at running actor.learn(), then save model, or exit if not at actor.learn()
                        self.save_ckpt(env_name=env.spec.id)
                        plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

                    s = s_new

                    if done:
                        break

                reward_buffer.append(ep_rs_sum)
                if plot_func is not None:
                    plot_func(reward_buffer)
                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                      .format(i_episode, train_episodes, ep_rs_sum, time.time() - t0))

                if i_episode % save_interval == 0:
                    self.save_ckpt(env_name=env.spec.id)
                    plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

            self.save_ckpt(env_name=env.spec.id)
            plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

        elif mode == 'test':
            self.load_ckpt(env_name=env.spec.id)
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))

            reward_buffer = []
            for i_episode in range(test_episodes):
                s = env.reset()
                ep_rs_sum = 0  # rewards of all steps
                for step in range(max_steps):
                    if render: env.render()
                    a = self.get_action_greedy(s)
                    s_new, r, done, info = env.step(a)
                    s_new = s_new

                    ep_rs_sum += r
                    s = s_new

                    if done:
                        break

                reward_buffer.append(ep_rs_sum)
                if plot_func:
                    plot_func(reward_buffer)
                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    i_episode, test_episodes, ep_rs_sum, time.time() - t0))

        elif mode is not 'test':
            print('unknow mode type')
