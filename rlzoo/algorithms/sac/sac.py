"""
Soft Actor-Critic
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
adding alpha loss
paper: https://arxiv.org/pdf/1812.05905.pdf
Actor policy is stochastic.
Env: Openai Gym Pendulum-v0, continuous action space
tensorflow 2.0.0a0
tensorflow-probability 0.6.0
tensorlayer 2.0.0
&&
pip install box2d box2d-kengz --user
"""

import time

import tensorflow_probability as tfp
import tensorlayer as tl
from rlzoo.common.utils import *
from rlzoo.common.buffer import *
from rlzoo.common.policy_networks import *
from rlzoo.common.value_networks import *

tfd = tfp.distributions
Normal = tfd.Normal

tl.logging.set_verbosity(tl.logging.DEBUG)


class SAC():
    """ Soft Actor-Critic """

    def __init__(self, net_list, optimizers_list, replay_buffer_capacity=5e5):
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.name = 'SAC'

        # get all networks
        [self.soft_q_net1, self.soft_q_net2, self.target_soft_q_net1, self.target_soft_q_net2,
         self.policy_net] = net_list

        assert isinstance(self.soft_q_net1, QNetwork)
        assert isinstance(self.soft_q_net2, QNetwork)
        assert isinstance(self.target_soft_q_net1, QNetwork)
        assert isinstance(self.target_soft_q_net2, QNetwork)
        assert isinstance(self.policy_net, StochasticPolicyNetwork)
        assert isinstance(self.policy_net.action_space, gym.spaces.Box)

        self.action_dim = self.policy_net.action_shape[0]

        self.log_alpha = tf.Variable(0, dtype=np.float32, name='log_alpha')
        self.alpha = tf.math.exp(self.log_alpha)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_soft_q_net1 = self.target_ini(self.soft_q_net1, self.target_soft_q_net1)
        self.target_soft_q_net2 = self.target_ini(self.soft_q_net2, self.target_soft_q_net2)

        [self.soft_q_optimizer1, self.soft_q_optimizer2, self.policy_optimizer, self.alpha_optimizer] = optimizers_list

    def evaluate(self, state, epsilon=1e-6):
        """ generate action with state for calculating gradients """
        _ = self.policy_net(state)
        mean, log_std = self.policy_net.policy_dist.get_param()  # as SAC uses TanhNorm instead of normal distribution, need original mean_std
        std = tf.math.exp(log_std)  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = tf.math.tanh(mean + std * z)  # TanhNormal distribution as actions; reparameterization trick
        # according to original paper, with an extra last term for normalizing different action range
        log_prob = Normal(mean, std).log_prob(mean + std * z) - tf.math.log(1. - action_0 ** 2 + epsilon)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the dim of actions to get 1 dim probability; or else use Multivariate Normal.
        log_prob = tf.reduce_sum(log_prob, axis=1)[:, np.newaxis]  # expand dim as reduce_sum causes 1 dim reduced

        action = action_0 * self.policy_net.policy_dist.action_scale + self.policy_net.policy_dist.action_mean

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        """ generate action with state for interaction with envronment """
        action, _, _, _, _ = self.evaluate(np.array([state]))
        return action.numpy()[0]

    def get_action_greedy(self, state):
        """ generate action with state for interaction with envronment """
        mean = self.policy_net(np.array([state]), greedy=True).numpy()[0]
        action = tf.math.tanh(mean) * self.policy_net.policy_dist.action_scale + self.policy_net.policy_dist.action_mean
        return action

    def sample_action(self, ):
        """ generate random actions for exploration """
        return self.policy_net.random_sample()

    def target_ini(self, net, target_net):
        """ hard-copy update for initializing target networks """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        """ update all networks in SAC """
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        reward = reward_scale * (reward -
                                 np.mean(reward, axis=0)) / (
                         np.std(reward, axis=0) + 1e-6)  # normalize with batch mean and std

        # Training Q Function
        new_next_action, next_log_prob, _, _, _ = self.evaluate(next_state)
        target_q_min = tf.minimum(
            self.target_soft_q_net1([next_state, new_next_action]),
            self.target_soft_q_net2([next_state, new_next_action])
        ) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.soft_q_net1([state, action])
            q_value_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value1, target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.soft_q_net1.trainable_weights)
        self.soft_q_optimizer1.apply_gradients(zip(q1_grad, self.soft_q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.soft_q_net2([state, action])
            q_value_loss2 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value2, target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.soft_q_net2.trainable_weights)
        self.soft_q_optimizer2.apply_gradients(zip(q2_grad, self.soft_q_net2.trainable_weights))

        # Training Policy Function
        with tf.GradientTape() as p_tape:
            new_action, log_prob, z, mean, log_std = self.evaluate(state)
            """ implementation 1 """
            predicted_new_q_value = tf.minimum(self.soft_q_net1([state, new_action]),
                                               self.soft_q_net2([state, new_action]))
            """ implementation 2 """
            # predicted_new_q_value = self.soft_q_net1([state, new_action])
            policy_loss = tf.reduce_mean(self.alpha * log_prob - predicted_new_q_value)
        p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

        # Updating alpha w.r.t entropy
        # alpha: trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean((self.log_alpha * (log_prob + target_entropy)))
            alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha = tf.math.exp(self.log_alpha)
        else:  # fixed alpha
            self.alpha = 1.
            alpha_loss = 0

        # Soft update the target value nets
        self.target_soft_q_net1 = self.target_soft_update(self.soft_q_net1, self.target_soft_q_net1, soft_tau)
        self.target_soft_q_net2 = self.target_soft_update(self.soft_q_net2, self.target_soft_q_net2, soft_tau)

    def save_ckpt(self, env_name):
        """ save trained weights """
        save_model(self.soft_q_net1, 'model_q_net1', self.name, env_name)
        save_model(self.soft_q_net2, 'model_q_net2', self.name, env_name)
        save_model(self.target_soft_q_net1, 'model_target_q_net1', self.name, env_name)
        save_model(self.target_soft_q_net2, 'model_target_q_net2', self.name, env_name)
        save_model(self.policy_net, 'model_policy_net', self.name, env_name)

    def load_ckpt(self, env_name):
        """ load trained weights """
        load_model(self.soft_q_net1, 'model_q_net1', self.name, env_name)
        load_model(self.soft_q_net2, 'model_q_net2', self.name, env_name)
        load_model(self.target_soft_q_net1, 'model_target_q_net1', self.name, env_name)
        load_model(self.target_soft_q_net2, 'model_target_q_net2', self.name, env_name)
        load_model(self.policy_net, 'model_policy_net', self.name, env_name)

    def learn(self, env, train_episodes=1000, test_episodes=1000, max_steps=150, batch_size=64, explore_steps=500,
              update_itr=3, policy_target_update_interval=3, reward_scale=1., save_interval=20,
              mode='train', AUTO_ENTROPY=True, render=False, plot_func=None):
        """
        :param env: learning environment
        :param train_episodes:  total number of episodes for training
        :param test_episodes:  total number of episodes for testing
        :param max_steps:  maximum number of steps for one episode
        :param batch_size:  udpate batchsize
        :param explore_steps:  for random action sampling in the beginning of training
        :param update_itr: repeated updates for single step
        :param policy_target_update_interval: delayed update for the policy network and target networks
        :param reward_scale: value range of reward
        :param save_interval: timesteps for saving the weights and plotting the results
        :param mode: 'train' or 'test'
        :param AUTO_ENTROPY: automatically updating variable alpha for entropy
        :param render: if true, visualize the environment
        :param plot_func: additional function for interactive module
        """

        # training loop
        if mode == 'train':
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            frame_idx = 0
            rewards = []
            t0 = time.time()
            for eps in range(train_episodes):
                state = env.reset()
                episode_reward = 0

                for step in range(max_steps):
                    if frame_idx > explore_steps:
                        action = self.get_action(state)
                    else:
                        action = self.sample_action()

                    next_state, reward, done, _ = env.step(action)
                    if render: env.render()
                    done = 1 if done == True else 0

                    self.replay_buffer.push(state, action, reward, next_state, done)

                    state = next_state
                    episode_reward += reward
                    frame_idx += 1

                    if len(self.replay_buffer) > batch_size:
                        for i in range(update_itr):
                            self.update(
                                batch_size, reward_scale=reward_scale, auto_entropy=AUTO_ENTROPY,
                                target_entropy=-1. * self.action_dim
                            )

                    if done:
                        break
                if eps % int(save_interval) == 0:
                    plot_save_log(rewards, algorithm_name=self.name, env_name=env.spec.id)
                    self.save_ckpt(env_name=env.spec.id)
                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                      .format(eps, train_episodes, episode_reward, time.time() - t0))
                rewards.append(episode_reward)
                if plot_func is not None:
                    plot_func(rewards)
            plot_save_log(rewards, algorithm_name=self.name, env_name=env.spec.id)
            self.save_ckpt(env_name=env.spec.id)

        elif mode == 'test':
            frame_idx = 0
            rewards = []
            t0 = time.time()
            self.load_ckpt(env_name=env.spec.id)
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            # set test mode
            self.soft_q_net1.eval()
            self.soft_q_net2.eval()
            self.target_soft_q_net1.eval()
            self.target_soft_q_net2.eval()
            self.policy_net.eval()

            for eps in range(test_episodes):
                state = env.reset()
                episode_reward = 0

                for step in range(max_steps):
                    action = self.get_action_greedy(state)
                    next_state, reward, done, _ = env.step(action)
                    if render: env.render()
                    done = 1 if done == True else 0

                    state = next_state
                    episode_reward += reward
                    frame_idx += 1
                    if done:
                        break
                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                      .format(eps, test_episodes, episode_reward, time.time() - t0))
                rewards.append(episode_reward)
            if plot_func:
                plot_func(rewards)

        else:
            print('unknow mode type')
