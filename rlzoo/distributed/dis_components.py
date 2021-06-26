import enum

import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import (barrier, request_variable,
                                   request_variable_with_template,
                                   save_variable, subset_all_reduce)
from kungfu.tensorflow.ops.queue import new_queue


class Role(enum.Enum):
    Learner = 1
    Actor = 2
    Server = 3


def show_role_name(role):
    return {
        Role.Learner: 'learner',
        Role.Actor: 'actor',
        Role.Server: 'server',
    }[role]


def _interval(n, offset=0):
    return list(range(offset, offset + n))


class Agent:
    def __init__(self, n_learners=1, n_actors=1, n_servers=1):
        rank = current_rank()
        size = current_cluster_size()
        if n_learners + n_actors + n_servers != size:
            raise RuntimeError('invalid cluster size')
        self._n_learners = n_learners
        self._n_actors = n_actors
        self._n_servers = n_servers
        self._global_rank = rank
        self._global_size = size
        roles = [Role.Learner] * n_learners + [Role.Actor] * n_actors + [Role.Server] * n_servers
        rank2role = dict(enumerate(roles))
        self._role = rank2role[rank]
        self._roles = {
            Role.Learner: _interval(n_learners),
            Role.Actor: _interval(n_actors, n_learners),
            Role.Server: _interval(n_servers, n_learners + n_actors),
        }
        self._role_sizes = {
            Role.Learner: n_learners,
            Role.Actor: n_actors,
            Role.Server: n_servers,
        }
        self._role_offsets = {
            Role.Learner: 0,
            Role.Actor: n_learners,
            Role.Server: n_learners + n_actors,
        }
        self._role_rank = self._global_rank - self._role_offsets[self._role]
        self._role_size = self._role_sizes[self._role]

    def _to_global_rank(self, role, role_rank):
        return int(self._role_offsets[role] + int(role_rank))

    # metadata APIs
    def role(self):
        return self._role

    def role_rank(self):
        return self._role_rank

    def role_size(self, role=None):
        if role is None:
            return self._role_size
        else:
            return self._role_sizes[role]

    # collective APIs
    def barrier(self):
        return barrier()

    def role_all_reduce(self, x):
        role_ranks = self._roles[self._role]
        topology = [i for i in range(self._global_size)]
        for i in role_ranks:
            topology[i] = role_ranks[0]
        # TODO: generate subset topology
        return subset_all_reduce(x, topology)

    # p2p APIs
    def save(self, x, name=None):
        return save_variable(x, name=name)

    def request(self, role: Role, role_rank, name, shape, dtype):
        role_size = self._role_sizes[role]
        assert (0 <= role_rank and role_rank < role_size)
        target = self._to_global_rank(role, role_rank)
        return request_variable(
            target,
            name=name,
            shape=shape,
            dtype=dtype,
        )

    def new_queue(self, src, dst):
        """create a uni-direction queue."""
        role1, rank1 = src
        role2, rank2 = dst
        srcRank = self._to_global_rank(role1, rank1)
        dstRank = self._to_global_rank(role2, rank2)
        return new_queue(srcRank, dstRank)

    def new_queue_pair(self, a, b):
        """create a pair of queues."""
        q1 = self.new_queue(a, b)
        q2 = self.new_queue(b, a)
        return q1, q2


class LearnerExample:
    pass


class ActorExample:
    pass


class ServerExample:
    pass
