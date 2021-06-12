import argparse
import json

import tensorflow as tf
from tensorflow.python.util import deprecation

import rlzoo

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-l', type=int, default=1)
    p.add_argument('-a', type=int, default=1)
    p.add_argument('-s', type=int, default=1)
    p.add_argument('-f', type=str, default='') # config.json
    args =  p.parse_args()
    if args.f:
        config = json.load(open(args.f))
    else:
        pass
    return args


def range1(n):
    return range(1, n + 1)


def train(agent):
    n_steps = 1
    for i in range1(n_steps):
        print('step %d' % (i))
        model = agent.request(rlzoo.Role.Server,
                              0,
                              'model',
                              shape=[1],
                              dtype=tf.float32)
        x = tf.Variable([10], dtype=tf.float32)
        y = agent.role_all_reduce(x)
        print(x)
        print(y)


def run_leaner(agent):
    agent.barrier()
    train(agent)


def run_actor(agent):
    agent.barrier()


def run_server(agent):
    model = tf.Variable([10], dtype=tf.float32)
    agent.save(model, name='model')
    print('saved')
    agent.barrier()  # save before barrier

def main():
    args = parse_args()
    agent = rlzoo.Agent(n_leaners=args.l, n_actors=args.a, n_servers=args.s)

    print('%s : %d/%d' % (agent.role(), agent.rank(), agent.size()))

    if agent.role() == rlzoo.Role.Leaner:
        run_leaner(agent)
    elif agent.role() == rlzoo.Role.Actor:
        run_actor(agent)
    elif agent.role() == rlzoo.Role.Server:
        run_server(agent)
    else:
        raise RuntimeError('invalid role')

    agent.barrier()


print('BEGIN')
main()
print('END')
