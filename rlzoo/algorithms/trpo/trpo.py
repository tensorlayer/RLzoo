"""
Trust Region Policy Optimization (TRPO)
---------------------------------------
PG method with a large step can collapse the policy performance,
even with a small step can lead a large differences in policy.
TRPO constraint the step in policy space using KL divergence (rather than in parameter space),
which can monotonically improve performance and avoid a collapsed update.

Reference
---------
Trust Region Policy Optimization, Schulman et al. 2015
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Approximately Optimal Approximate Reinforcement Learning, Kakade and Langford 2002
openai/spinningup : http://spinningup.openai.com/en/latest/algorithms/trpo.html

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

"""

Trust Region Policy Optimization 

(with support for Natural Policy Gradient)

"""


class TRPO:
    """
    trpo class
    """

    def __init__(self, net_list, optimizers_list, damping_coeff=0.1, cg_iters=10, delta=0.01):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param damping_coeff: Artifact for numerical stability
        :param cg_iters: Number of iterations of conjugate gradient to perform
        :param delta: KL-divergence limit for TRPO update.
        """

        assert len(net_list) == 2
        assert len(optimizers_list) == 1

        self.name = 'TRPO'

        self.critic, self.actor = net_list

        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

        self.damping_coeff, self.cg_iters = damping_coeff, cg_iters
        self.delta = delta

        # Optimizer for value function
        self.critic_opt, = optimizers_list
        self.old_dist = make_dist(self.actor.action_space)

    @staticmethod
    def flat_concat(xs):
        """
        flat concat input

        :param xs: a list of tensor

        :return: flat tensor
        """
        return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

    @staticmethod
    def assign_params_from_flat(x, params):
        """
        assign params from flat input

        :param x:
        :param params:

        :return: group
        """
        flat_size = lambda p: int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
        return tf.group([p.assign(p_new) for p, p_new in zip(params, new_params)])

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

    def get_v(self, s):
        """
        Compute value

        :param s: state

        :return: value
        """
        if s.ndim < 2: s = s[np.newaxis, :]
        res = self.critic(s)[0, 0]
        return res

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

    # TRPO losses
    def pi_loss(self, inputs):
        """
        calculate pi loss

        :param inputs: a list of x_ph, a_ph, adv_ph, ret_ph, logp_old_ph and other inputs

        :return: pi loss
        """
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs

        pi, logp, logp_pi, info, info_phs, d_kl = self.actor.cal_outputs_1(x_ph, a_ph, *info_values)
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        pi_loss = -tf.reduce_mean(ratio * adv_ph)
        return pi_loss

    # Symbols needed for CG solver
    def gradient(self, inputs):
        """
        pi gradients

        :param inputs: a list of x_ph, a_ph, adv_ph, ret_ph, logp_old_ph and other inputs

        :return: gradient
        """
        pi_params = self.actor.trainable_weights
        with tf.GradientTape() as tape:
            loss = self.pi_loss(inputs)
        grad = tape.gradient(loss, pi_params)
        grad = self.flat_concat(grad)
        return grad

    # Symbols for getting and setting params
    def get_pi_params(self):
        """
        get actor trainable parameters

        :return: flat actor trainable parameters
        """
        pi_params = self.actor.trainable_weights
        return self.flat_concat(pi_params)

    def set_pi_params(self, v_ph):
        """
        set actor trainable parameters

        :param v_ph: inputs

        :return: None
        """
        pi_params = self.actor.trainable_weights
        self.assign_params_from_flat(v_ph, pi_params)

    def cg(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = copy.deepcopy(b)  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)

        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def eval(self, bs, ba, badv, oldpi_prob):
        _ = self.actor(bs)
        pi_prob = tf.exp(self.actor.policy_dist.logp(ba))
        ratio = pi_prob / (oldpi_prob + EPS)

        surr = ratio * badv
        aloss = -tf.reduce_mean(surr)
        kl = self.old_dist.kl(self.actor.policy_dist.param)
        # kl = tfp.distributions.kl_divergence(oldpi, pi)
        kl = tf.reduce_mean(kl)
        return aloss, kl

    def a_train(self, s, a, adv, oldpi_prob, backtrack_iters, backtrack_coeff):
        s = np.array(s)
        a = np.array(a, np.float32)
        adv = np.array(adv, np.float32)

        with tf.GradientTape() as tape:
            aloss, kl = self.eval(s, a, adv, oldpi_prob)
        a_grad = tape.gradient(aloss, self.actor.trainable_weights)
        # print(a_grad)
        a_grad = self.flat_concat(a_grad)
        pi_l_old = aloss
        # print(a_grad)

        Hx = lambda x: self.hessian_vector_product(s, a, adv, oldpi_prob, x)
        x = self.cg(Hx, a_grad)
        alpha = np.sqrt(2 * self.delta / (np.dot(x, Hx(x)) + EPS))

        old_params = self.flat_concat(self.actor.trainable_weights)

        for j in range(backtrack_iters):
            self.set_pi_params(old_params - alpha * x * backtrack_coeff ** j)
            kl, pi_l_new = self.eval(s, a, adv, oldpi_prob)
            if kl <= self.delta and pi_l_new <= pi_l_old:
                # Accepting new params at step j of line search.
                break

            if j == backtrack_iters - 1:
                # Line search failed! Keeping old params.
                self.set_pi_params(old_params)

    def hessian_vector_product(self, s, a, adv, oldpi_prob, v_ph):
        # for H = grad**2 f, compute Hx
        params = self.actor.trainable_weights

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                aloss, kl = self.eval(s, a, adv, oldpi_prob)
            g = tape0.gradient(kl, params)
            g = self.flat_concat(g)
            assert v_ph.shape == g.shape
            v = tf.reduce_sum(g * v_ph)
        grad = tape1.gradient(v, params)
        hvp = self.flat_concat(grad)

        if self.damping_coeff > 0:
            hvp += self.damping_coeff * v_ph
        return hvp

    def update(self, bs, ba, br, train_critic_iters, backtrack_iters, backtrack_coeff):
        """
        update trpo

        :return: None
        """
        adv = self.cal_adv(bs, br)
        _ = self.actor(bs)
        oldpi_prob = tf.exp(self.actor.policy_dist.logp(ba))
        oldpi_prob = tf.stop_gradient(oldpi_prob)

        oldpi_param = self.actor.policy_dist.get_param()
        self.old_dist.set_param(oldpi_param)

        self.a_train(bs, ba, adv, oldpi_prob, backtrack_iters, backtrack_coeff)

        for _ in range(train_critic_iters):
            self.c_train(br, bs)

    def learn(self, env, train_episodes=200, test_episodes=100, max_steps=200, save_interval=10,
              gamma=0.9, mode='train', render=False, batch_size=32, backtrack_iters=10, backtrack_coeff=0.8,
              train_critic_iters=80, plot_func=None):
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
        :param backtrack_iters: Maximum number of steps allowed in the backtracking line search
        :param backtrack_coeff: How far back to step during backtracking line search
        :param train_critic_iters: critic update iteration steps
        
        :return: None
        """

        t0 = time.time()

        if mode == 'train':
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            for ep in range(1, train_episodes + 1):
                s = env.reset()
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_rs_sum = 0
                for t in range(max_steps):  # in one episode
                    if render:
                        env.render()
                    a = self.get_action(s)

                    s_, r, done, _ = env.step(a)
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)
                    s = s_
                    ep_rs_sum += r

                    # update ppo
                    if (t + 1) % batch_size == 0 or t == max_steps - 1 or done:
                        if done:
                            v_s_ = 0
                        else:
                            try:
                                v_s_ = self.get_v(s_)
                            except:
                                v_s_ = self.get_v(s_[np.newaxis, :])  # for raw-pixel input
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()
                        bs = buffer_s
                        ba, br = buffer_a, np.array(discounted_r)[:, np.newaxis]
                        buffer_s, buffer_a, buffer_r = [], [], []
                        self.update(bs, ba, br, train_critic_iters, backtrack_iters, backtrack_coeff)
                    if done:
                        break

                print(
                    'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        ep, train_episodes, ep_rs_sum,
                        time.time() - t0
                    )
                )

                reward_buffer.append(ep_rs_sum)
                if plot_func is not None:
                    plot_func(reward_buffer)
                if ep and not ep % save_interval:
                    self.save_ckpt(env_name=env.spec.id)
                    plot_save_log(reward_buffer, self.name, env.spec.id)

            self.save_ckpt(env_name=env.spec.id)
            plot_save_log(reward_buffer, self.name, env.spec.id)

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
