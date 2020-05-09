"""
Twin Delayed DDPG (TD3)
------------------------
DDPG suffers from problems like overestimate of Q-values and sensitivity to hyper-parameters.
Twin Delayed DDPG (TD3) is a variant of DDPG with several tricks:
* Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), 
and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

* Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently 
than the Q-function. 

* Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for 
the policy to exploit Q-function errors by smoothing out Q along changes in action.

The implementation of TD3 includes 6 networks: 2 Q-net, 2 target Q-net, 1 policy net, 1 target policy net
Actor policy in TD3 is deterministic, with Gaussian exploration noise.

Reference
---------
original paper: https://arxiv.org/pdf/1802.09477.pdf


Environment
---
Openai Gym Pendulum-v0, continuous action space
https://gym.openai.com/envs/Pendulum-v0/

Prerequisites
---
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

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


###############################  TD3  ####################################


class TD3():
    """ twin-delayed ddpg """

    def __init__(self, net_list, optimizers_list, replay_buffer_capacity=5e5, policy_target_update_interval=5):
        self.name = 'TD3'
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

        # get all networks
        [self.q_net1, self.q_net2, self.target_q_net1, self.target_q_net2, self.policy_net,
         self.target_policy_net] = net_list

        assert isinstance(self.q_net1, QNetwork)
        assert isinstance(self.q_net2, QNetwork)
        assert isinstance(self.target_q_net1, QNetwork)
        assert isinstance(self.target_q_net2, QNetwork)
        assert isinstance(self.policy_net, DeterministicPolicyNetwork)
        assert isinstance(self.target_policy_net, DeterministicPolicyNetwork)
        assert isinstance(self.policy_net.action_space, gym.spaces.Box)

        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        [self.q_optimizer1, self.q_optimizer2, self.policy_optimizer] = optimizers_list

    def evaluate(self, state, eval_noise_scale, target=False):
        """
        generate action with state for calculating gradients;

        :param eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        """
        if target:
            action = self.target_policy_net(state)
        else:
            action = self.policy_net(state)
        # add noise
        normal = Normal(0, 1)
        eval_noise_clip = 2 * eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
        action = action + noise

        return action

    def get_action(self, state, explore_noise_scale):
        """ generate action with state for interaction with envronment """
        action = self.policy_net(np.array([state]))
        action = action.numpy()[0]

        # add noise
        normal = Normal(0, 1)
        noise = normal.sample(action.shape) * explore_noise_scale
        action = action + noise

        return action.numpy()

    def get_action_greedy(self, state):
        """ generate action with state for interaction with envronment """
        return self.policy_net(np.array([state])).numpy()[0]

    def sample_action(self):
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

    def update(self, batch_size, eval_noise_scale, reward_scale=1., gamma=0.9, soft_tau=1e-2):
        """ update all networks in TD3 """
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        new_next_action = self.evaluate(
            next_state, eval_noise_scale=eval_noise_scale, target=True
        )  # clipped normal noise
        reward = reward_scale * (reward -
                                 np.mean(reward, axis=0)) / (np.std(reward,
                                                                    axis=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = tf.minimum(self.target_q_net1([next_state, new_next_action]),
                                  self.target_q_net2([next_state, new_next_action]))

        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1([state, action])
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2([state, action])
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            with tf.GradientTape() as p_tape:
                new_action = self.evaluate(
                    state, eval_noise_scale=0.0, target=False
                )  # no noise, deterministic policy gradients
                # """ implementation 1 """
                # predicted_new_q_value = tf.minimum(self.q_net1([state, new_action]),self.q_net2([state, new_action]))
                """ implementation 2 """
                predicted_new_q_value = self.q_net1([state, new_action])
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_ckpt(self, env_name):  # save trained weights
        save_model(self.q_net1, 'model_q_net1', self.name, env_name)
        save_model(self.q_net2, 'model_q_net2', self.name, env_name)
        save_model(self.target_q_net1, 'model_target_q_net1', self.name, env_name)
        save_model(self.target_q_net2, 'model_target_q_net2', self.name, env_name)
        save_model(self.policy_net, 'model_policy_net', self.name, env_name)
        save_model(self.target_policy_net, 'model_target_policy_net', self.name, env_name)

    def load_ckpt(self, env_name):  # load trained weights
        load_model(self.q_net1, 'model_q_net1', self.name, env_name)
        load_model(self.q_net2, 'model_q_net2', self.name, env_name)
        load_model(self.target_q_net1, 'model_target_q_net1', self.name, env_name)
        load_model(self.target_q_net2, 'model_target_q_net2', self.name, env_name)
        load_model(self.policy_net, 'model_policy_net', self.name, env_name)
        load_model(self.target_policy_net, 'model_target_policy_net', self.name, env_name)

    def learn(self, env, train_episodes=1000, test_episodes=1000, max_steps=150, batch_size=64, explore_steps=500,
              update_itr=3,
              reward_scale=1., save_interval=10, explore_noise_scale=1.0, eval_noise_scale=0.5, mode='train',
              render=False, plot_func=None):
        """
        :param env: learning environment
        :param train_episodes:  total number of episodes for training
        :param test_episodes:  total number of episodes for testing
        :param max_steps:  maximum number of steps for one episode
        :param batch_size:  udpate batchsize
        :param explore_steps:  for random action sampling in the beginning of training
        :param update_itr: repeated updates for single step
        :param reward_scale: value range of reward
        :param save_interval: timesteps for saving the weights and plotting the results
        :param explore_noise_scale: range of action noise for exploration
        :param eval_noise_scale: range of action noise for evaluation of action value
        :param mode: 'train' or 'test'
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
                        action = self.get_action(state, explore_noise_scale=explore_noise_scale)
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
                            self.update(batch_size, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)

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
            self.q_net1.eval()
            self.q_net2.eval()
            self.target_q_net1.eval()
            self.target_q_net2.eval()
            self.policy_net.eval()
            self.target_policy_net.eval()

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
                if plot_func is not None:
                    plot_func(rewards)

        else:
            print('unknow mode type, activate test mode as default')
