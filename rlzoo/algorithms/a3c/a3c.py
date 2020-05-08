"""
Asynchronous Advantage Actor Critic (A3C) with Continuous Action Space.

Actor Critic History
----------------------
A3C > DDPG (for continuous action space) > AC

Advantage
----------
Train faster and more stable than AC.

Disadvantage
-------------
Have bias.

Reference
----------
Original Paper: https://arxiv.org/pdf/1602.01783.pdf
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/
Environment
-----------
BipedalWalker-v2 : https://gym.openai.com/envs/BipedalWalker-v2

Reward is given for moving forward, total 300+ points up to the far end.
If the robot falls, it gets -100. Applying motor torque costs a small amount of
points, more optimal agent will get better score. State consists of hull angle
speed, angular velocity, horizontal speed, vertical speed, position of joints
and joints angular speed, legs contact with ground, and 10 lidar rangefinder
measurements. There's no coordinates in the state vector.

Prerequisites
--------------
tensorflow 2.0.0a0
tensorflow-probability 0.6.0
tensorlayer 2.0.0
&&
pip install box2d box2d-kengz --user

"""

import multiprocessing
import threading
import time

from rlzoo.common.utils import *
from rlzoo.common.buffer import *


# tl.logging.set_verbosity(tl.logging.DEBUG)
###################  Asynchronous Advantage Actor Critic (A3C)  ####################################
class ACNet(object):

    def __init__(self, net_list, scope, entropy_beta):
        self.ENTROPY_BETA = entropy_beta
        self.actor, self.critic = net_list

    # @tf.function  # shouldn't use here!
    def update_global(
            self, buffer_s, buffer_a, buffer_v_target, globalAC
    ):  # refer to the global Actor-Crtic network for updating it with samples
        """ update the global critic """
        with tf.GradientTape() as tape:
            self.v = self.critic(buffer_s)
            self.v_target = buffer_v_target
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            self.c_loss = tf.reduce_mean(tf.square(td))
        self.c_grads = tape.gradient(self.c_loss, self.critic.trainable_weights)
        OPT_C.apply_gradients(zip(self.c_grads, globalAC.critic.trainable_weights))  # local grads applies to global net
        del tape  # Drop the reference to the tape
        """ update the global actor """
        with tf.GradientTape() as tape:
            self.actor(buffer_s)
            self.a_his = buffer_a  # float32
            log_prob = self.actor.policy_dist.logp(self.a_his)
            exp_v = log_prob * td  # td is from the critic part, no gradients for it
            entropy = self.actor.policy_dist.entropy()  # encourage exploration
            self.exp_v = self.ENTROPY_BETA * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
        self.a_grads = tape.gradient(self.a_loss, self.actor.trainable_weights)
        OPT_A.apply_gradients(zip(self.a_grads, globalAC.actor.trainable_weights))  # local grads applies to global net
        del tape  # Drop the reference to the tape

    # @tf.function
    def pull_global(self, globalAC):  # run by a local, pull weights from the global nets
        for l_p, g_p in zip(self.actor.trainable_weights, globalAC.actor.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.critic.trainable_weights, globalAC.critic.trainable_weights):
            l_p.assign(g_p)

    def get_action(self, s):  # run by a local
        return self.actor(np.array([s])).numpy()[0]

    def get_action_greedy(self, s):
        return self.actor(np.array([s]), greedy=True)[0].numpy()

    def save_ckpt(self, env_name):  # save trained weights
        save_model(self.actor, 'model_actor', 'A3C', env_name)
        save_model(self.critic, 'model_critic', 'A3C', env_name)

    def load_ckpt(self, env_name):  # load trained weights
        load_model(self.actor, 'model_actor', 'A3C', env_name)
        load_model(self.critic, 'model_critic', 'A3C', env_name)


class Worker(object):
    def __init__(self, env, net_list, name, train_episodes, max_steps, gamma, update_itr, entropy_beta,
                 render, plot_func):
        self.name = name
        self.AC = ACNet(net_list, name, entropy_beta)
        self.MAX_GLOBAL_EP = train_episodes
        self.UPDATE_GLOBAL_ITER = update_itr
        self.GAMMA = gamma
        self.env = env
        self.max_steps = max_steps
        self.render = render
        self.plot_func = plot_func

    def work(self, globalAC):
        global COORD, GLOBAL_RUNNING_R, GLOBAL_EP, OPT_A, OPT_C, t0, SAVE_INTERVAL
        total_step = 1
        save_cnt = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < self.MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for epi_step in range(self.max_steps):
                # visualize Worker_0 during training
                if self.name == 'Worker_0' and total_step % 30 == 0 and self.render:
                    self.env.render()
                s = s.astype('float32')  # double to float
                a = self.AC.get_action(s)
                s_, r, done, _info = self.env.step(a)

                s_ = s_.astype('float32')  # double to float

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % self.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net

                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.AC.critic(s_[np.newaxis, :])[0, 0]  # reduce dim from 2 to 0

                    buffer_v_target = []

                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.GAMMA * v_s_
                        buffer_v_target.append(v_s_)

                    buffer_v_target.reverse()
                    buffer_s = buffer_s if len(buffer_s[0].shape) > 1 else np.vstack(
                        buffer_s)  # no vstack for raw-pixel input
                    buffer_a, buffer_v_target = (
                        np.vstack(buffer_a), np.vstack(buffer_v_target)
                    )

                    # update gradients on global network
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target.astype('float32'), globalAC)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # update local network from global network
                    self.AC.pull_global(globalAC)

                s = s_
                total_step += 1
                if self.name == 'Worker_0' and GLOBAL_EP >= save_cnt * SAVE_INTERVAL:
                    plot_save_log(GLOBAL_RUNNING_R, algorithm_name=self.name, env_name=self.env.spec.id)
                    globalAC.save_ckpt(env_name=self.env.spec.id)
                    save_cnt += 1
                if done:
                    break

            GLOBAL_RUNNING_R.append(ep_r)
            if self.name == 'Worker_0' and self.plot_func is not None:
                self.plot_func(GLOBAL_RUNNING_R)
            print('{}, Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                  .format(self.name, GLOBAL_EP, self.MAX_GLOBAL_EP, ep_r, time.time() - t0))
            GLOBAL_EP += 1


class A3C():
    def __init__(self, net_list, optimizers_list, entropy_beta=0.005):
        """
        :param entropy_beta: factor for entropy boosted exploration
        """
        self.net_list = net_list
        self.optimizers_list = optimizers_list
        self.GLOBAL_AC = ACNet(self.net_list[0], 'global', entropy_beta)  # we only need its params
        self.entropy_beta = entropy_beta
        self.name = 'A3C'

    def learn(self, env, train_episodes=1000, test_episodes=10, max_steps=150, render=False, n_workers=1, update_itr=10,
              gamma=0.99, save_interval=500, mode='train', plot_func=None):

        """
        :param env: a list of same learning environments
        :param train_episodes:  total number of episodes for training
        :param test_episodes:  total number of episodes for testing
        :param max_steps:  maximum number of steps for one episode
        :param render: render or not
        :param n_workers: manually set number of workers
        :param update_itr: update global policy after several episodes
        :param gamma: reward discount factor
        :param save_interval: timesteps for saving the weights and plotting the results
        :param mode: train or test
        :param plot_func: additional function for interactive module
        """
        global COORD, GLOBAL_RUNNING_R, GLOBAL_EP, OPT_A, OPT_C, t0, SAVE_INTERVAL
        SAVE_INTERVAL = save_interval
        COORD = tf.train.Coordinator()
        GLOBAL_RUNNING_R = []
        GLOBAL_EP = 0  # will increase during training, stop training when it >= MAX_GLOBAL_EP
        N_WORKERS = n_workers if n_workers > 0 else multiprocessing.cpu_count()

        self.plot_func = plot_func
        if mode == 'train':
            # ============================= TRAINING ===============================
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env[0].spec.id))
            t0 = time.time()
            with tf.device("/cpu:0"):
                [OPT_A, OPT_C] = self.optimizers_list

                workers = []
                # Create worker
                for i in range(N_WORKERS):
                    i_name = 'Worker_%i' % i  # worker name
                    workers.append(
                        Worker(env[i], self.net_list[i + 1], i_name, train_episodes, max_steps, gamma,
                               update_itr, self.entropy_beta, render, plot_func))

            # start TF threading
            worker_threads = []
            for worker in workers:
                # t = threading.Thread(target=worker.work)
                job = lambda: worker.work(self.GLOBAL_AC)
                t = threading.Thread(target=job)
                t.start()
                worker_threads.append(t)

            COORD.join(worker_threads)

            plot_save_log(GLOBAL_RUNNING_R, algorithm_name=self.name, env_name=env[0].spec.id)
            self.GLOBAL_AC.save_ckpt(env_name=env[0].spec.id)

        elif mode == 'test':
            # ============================= EVALUATION =============================
            env = env[0]  # only need one env for test
            self.GLOBAL_AC.load_ckpt(env_name=env.spec.id)
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            frame_idx = 0
            for eps in range(test_episodes):
                s = env.reset()
                rall = 0
                for step in range(max_steps):
                    env.render()
                    frame_idx += 1
                    s = s.astype('float32')  # double to float
                    a = self.GLOBAL_AC.get_action_greedy(s)
                    s, r, d, _ = env.step(a)
                    if render:
                        env.render()
                    rall += r
                    if d:
                        break

                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    eps, test_episodes, rall, time.time() - t0))

        elif mode is not 'test':
            print('unknow mode type')
