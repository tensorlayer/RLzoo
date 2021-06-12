import argparse
import json

# import tensorflow as tf
# from tensorflow.python.util import deprecation

import rlzoo
import tensorflow as tf

# deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-l', type=int, default=1)
    p.add_argument('-a', type=int, default=1)
    p.add_argument('-s', type=int, default=1)
    p.add_argument('-f', type=str, default='') # config.json

    p.add_argument('-step', type=int, default=10)

    args =  p.parse_args()
    # if args.f:
    #     config = json.load(open(args.f))
    # else:
    #     pass
    return args


def range1(n):
    return range(1, n + 1)


def train(agent):
    n_steps = 1
    for i in range1(n_steps):
        print('step %d' % (i))
        # model = agent.request(rlzoo.Role.Server,
        #                       0,
        #                       'model',
        #                       shape=[1],
        #                       dtype=tf.float32)
        # x = tf.Variable([10], dtype=tf.float32)
        # y = agent.role_all_reduce(x)
        # print(x)
        # print(y)


def run_leaner(agent, args):
    agent.barrier()
    train(agent)


# action :: server -> actor
# state :: actor -> server

def run_actor(agent, args): # sampler
    agent.barrier()

    action_q, state_q = agent.new_queue_pair((rlzoo.Role.Server, 0), (rlzoo.Role.Actor, agent.role_rank()))


    for i in range(args.step):
        print('step %d' % (i))
        a = action_q.get(dtype=tf.int32, shape=(1,))
        print('action: %s' % (a))

        s = tf.Variable([1,2,3], dtype=tf.int32)
        state_q.put(s)

        r = tf.Variable([1.0], dtype=tf.float32)
        state_q.put(r)


def run_server(agent, args):
    # model = tf.Variable([10], dtype=tf.float32)
    # agent.save(model, name='model')
    # print('saved')
    agent.barrier()  # save before barrier

    # create queue after barrier
    qs = [agent.new_queue_pair((rlzoo.Role.Server, 0), (rlzoo.Role.Actor, i)) for i in range(agent.role_size(rlzoo.Role.Actor))]
    print(qs)
    action_qs, state_qs = zip(*qs)

    for i in range(args.step):
        print('step %d' % (i))
        for i, aq in enumerate(action_qs):
            a = tf.Variable([i], dtype=tf.int32)
            aq.put(a)

        for i, sq in enumerate(state_qs):
            s = sq.get(dtype=tf.int32, shape=(3,))
            print('state from %d: %s' % (i, s))
            r = sq.get(dtype=tf.float32, shape=(1,))
            print('reward from %d: %s' % (i, r))


def main():
    args = parse_args()
    agent = rlzoo.Agent(n_leaners=args.l, n_actors=args.a, n_servers=args.s)

    print('%s : %d/%d' % (agent.role(), agent.role_rank(), agent.role_size()))

    if agent.role() == rlzoo.Role.Leaner:
        run_leaner(agent, args)
    elif agent.role() == rlzoo.Role.Actor:
        run_actor(agent, args)
    elif agent.role() == rlzoo.Role.Server:
        run_server(agent, args)
    else:
        raise RuntimeError('invalid role')

    agent.barrier()


print('BEGIN')
main()
print('END')
