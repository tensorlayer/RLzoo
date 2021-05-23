from rlzoo.common.policy_networks import StochasticPolicyNetwork
from rlzoo.common.value_networks import ValueNetwork
import numpy as np
from rlzoo.common.utils import *
import pickle


class DPPOGlobalManager:
    def __init__(self, net_builder, opt_builder, param_pipe_list, name='DPPO_CLIP'):
        networks = net_builder()
        optimizers_list = opt_builder()
        assert len(networks) == 2
        assert len(optimizers_list) == 2
        self.critic, self.actor = networks
        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)
        self.critic_opt, self.actor_opt = optimizers_list
        self.param_pipe_list = param_pipe_list
        self.name = name

    def run(self, traj_queue, grad_queue, should_stop, should_update, barrier,
            max_update_num=1000, update_interval=100, save_interval=10, env_name='CartPole-v0'):
        update_cnt = 0
        while update_cnt < max_update_num:
            batch_a_grad, batch_c_grad = [], []
            for _ in range(update_interval):
                a_grad, c_grad = grad_queue.get()
                batch_a_grad.append(a_grad)
                batch_c_grad.append(c_grad)

            # update
            should_update.set()
            self.update_model(batch_a_grad, batch_c_grad)
            self.send_param()

            traj_queue.empty()
            for q in grad_queue: q.empty()

            barrier.wait()
            should_update.clear()

            update_cnt += 1
            if update_cnt // save_interval == 0:
                self.save_model(env_name)
        should_stop.set()

    def send_param(self):
        params = self.critic.trainable_weights + self.actor.trainable_weights
        for pipe_connection in self.param_pipe_list:
            pipe_connection.send(params)

    def update_model(self, batch_a_grad, batch_c_grad):
        a_grad = np.mean(batch_a_grad, axis=0)
        c_grad = np.mean(batch_c_grad, axis=0)
        self.actor_opt.apply_gradients(zip(a_grad, self.actor.trainable_weights))
        self.critic_opt.apply_gradients(zip(c_grad, self.critic.trainable_weights))

    def save_model(self, env_name):
        save_model(self.actor, 'actor', self.name, env_name)
        save_model(self.critic, 'critic', self.name, env_name)

    # todo load model
