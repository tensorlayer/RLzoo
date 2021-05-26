from rlzoo.common.policy_networks import StochasticPolicyNetwork
from rlzoo.common.value_networks import ValueNetwork
from rlzoo.common.utils import *
import queue


def write_log(text: str):
    pass
    # print('global manager: '+text)
    # with open('global_manager_log.txt', 'a') as f:
    #     f.write(str(text) + '\n')


class DPPOGlobalManager:
    def __init__(self, net_builder, opt_builder, name='DPPO_CLIP'):
        self.net_builder, self.opt_builder = net_builder, opt_builder
        self.name = name
        self.critic, self.actor = None, None
        self.critic_opt, self.actor_opt = None, None

    def init_components(self):
        networks = self.net_builder()
        optimizers_list = self.opt_builder()
        assert len(networks) == 2
        assert len(optimizers_list) == 2
        self.critic, self.actor = networks
        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)
        self.critic_opt, self.actor_opt = optimizers_list

    def run(self, traj_queue, grad_queue, should_stop, should_update, barrier, param_pipe_list,
            max_update_num=1000, update_interval=100, save_interval=10, env_name='CartPole-v0'):

        self.init_components()

        if should_update.is_set():
            write_log('syn model')
            self.send_param(param_pipe_list)
            write_log('wait for barrier')
            barrier.wait()
            should_update.clear()

        update_cnt = 0
        batch_a_grad, batch_c_grad = [], []
        while update_cnt < max_update_num:
            # print('\rupdate cnt {}, traj_que {}, grad_que {}'.format(
            #     update_cnt, traj_queue.qsize(), grad_queue[0].qsize()), end='')
            print('update cnt {}, traj_que {}, grad_que {}'.format(
                update_cnt, traj_queue.qsize(), grad_queue[0].qsize()))
            try:
                a_grad, c_grad = [q.get(timeout=1) for q in grad_queue]
                batch_a_grad.append(a_grad)
                batch_c_grad.append(c_grad)
                write_log('got grad')
            except queue.Empty:
                continue

            if len(batch_a_grad) > update_interval and len(batch_c_grad) > update_interval:
                # write_log('ready to update')
                # update
                should_update.set()
                write_log('update model')
                self.update_model(batch_a_grad, batch_c_grad)
                write_log('send_param')
                self.send_param(param_pipe_list)

                write_log('empty queue')
                traj_queue.empty()
                for q in grad_queue:
                    q.empty()
                batch_a_grad.clear()
                batch_c_grad.clear()

                write_log('wait for barrier')
                barrier.wait()
                should_update.clear()
                barrier.reset()
                update_cnt += 1
                if update_cnt // save_interval == 0:
                    self.save_model(env_name)
        should_stop.set()

    def send_param(self, param_pipe_list):
        params = self.critic.trainable_weights + self.actor.trainable_weights
        params = [p.numpy() for p in params]
        for i, pipe_connection in enumerate(param_pipe_list):
            pipe_connection.put(params)

    def update_model(self, batch_a_grad, batch_c_grad):
        a_grad = np.mean(batch_a_grad, axis=0)
        c_grad = np.mean(batch_c_grad, axis=0)
        self.actor_opt.apply_gradients(zip(a_grad, self.actor.trainable_weights))
        self.critic_opt.apply_gradients(zip(c_grad, self.critic.trainable_weights))

    def save_model(self, env_name):
        save_model(self.actor, 'actor', self.name, env_name)
        save_model(self.critic, 'critic', self.name, env_name)

    # todo load model
