"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import inspect
import operator
import random

import numpy as np


class ReplayBuffer(object):
    """A standard ring buffer for storing transitions and sampling for training"""
    def __init__(self, capacity):
        self.capacity = capacity  # mamimum number of samples
        self.buffer = []
        self.position = 0  # pointer

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        indexes = range(len(self))
        # sample with replacement
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxes:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            np.stack(states),
            np.stack(actions),
            np.stack(rewards),
            np.stack(next_states),
            np.stack(dones),
        )

    def __len__(self):
        return len(self.buffer)


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        :param apacity: (int)
            Total size of the array - must be a power of two.
        :param operation: (lambda obj, obj -> obj)
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        :param neutral_element: (obj)
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences

        Returns:
            reduced: (obj) result of reducing self.operation over the specified range of array.
        """
        if end is None:
            end = self._capacity - 1
        if end < 0:
            end += self._capacity
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        :param perfixsum: (float)
            upperbound on the sum of array prefix

        Returns:
            idx: (int)
                highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class PrioritizedReplayBuffer(ReplayBuffer):  # is it succeed from the ReplayBuffer above?
    def __init__(self, capacity, alpha, beta):
        """Create Prioritized Replay buffer.

        :param capacity: (int)
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        :param alpha: (float)
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also:
            ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.beta = beta

    def push(self, *args):
        """See ReplayBuffer.store_effect"""
        idx = self.position
        super().push(*args)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.buffer) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        idxes = self._sample_proportional(batch_size)

        it_sum = self._it_sum.sum()
        p_min = self._it_min.min() / it_sum
        max_weight = (p_min * len(self.buffer))**(-self.beta)

        p_samples = np.asarray([self._it_sum[idx] for idx in idxes]) / it_sum
        weights = (p_samples * len(self.buffer)) ** (-self.beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample + (weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class HindsightReplayBuffer(ReplayBuffer):
    """Hindsight Experience Replay
    In this buffer, state is a tuple consists of (observation, goal)
    """
    GOAL_FUTURE = 'future'
    GOAL_EPISODE = 'episode'
    GOAL_RANDOM = 'random'

    def __init__(self, capacity, hindsight_freq, goal_type, reward_func, done_func):
        """
        :param hindsight_freq (int): How many hindsight transitions will be generated for each real transition
        :param goal_type (str): The generatation method of hindsight goals. Should be HER_GOAL_*
        :param reward_func (callable): goal (np.array) X next_state (np.array) -> reward (float)
        :param done_func (callable): goal (np.array) X next_state (np.array) -> done_flag (bool)
        """
        super().__init__(capacity)
        self.hindsight_freq = hindsight_freq
        self.goal_type = goal_type
        self.reward_func = reward_func
        self.done_func = done_func

    def _sample_goals(self, episode, t):
        goals = []
        episode_len = len(episode)
        for _ in range(self.hindsight_freq):
            if self.goal_type == HindsightReplayBuffer.GOAL_FUTURE:
                index = random.choice(range(t + 1, episode_len))
                source = episode
            elif self.goal_type == HindsightReplayBuffer.GOAL_EPISODE:
                index = random.choice(range(episode_len))
                source = episode
            elif self.goal_type == HindsightReplayBuffer.GOAL_RANDOM:
                index = random.choice(range(len(self)))
                source = self.buffer
            else:
                raise ValueError("Invalid goal type %s" % self.goal_type)
            goals.append(source[index][0][0])  # return the observation
        return goals

    def push(self, *args, **kwargs):
        if inspect.stack()[1][3] != 'push_episode':
            raise ValueError("Please use `push_episode` methods in HER")
        else:
            super().push(*args, **kwargs)

    def push_episode(self, states, actions, rewards, next_states, dones):
        episode = list(zip(states, actions, rewards, next_states, dones))
        episode_len = len(states)
        for t, (state, action, reward, next_state, done) in enumerate(episode):
            self.push(state, action, reward, next_state, done)
            if self.goal_type == HindsightReplayBuffer.GOAL_FUTURE and t == episode_len - 1:
                break
            for goal in self._sample_goals(episode, t):
                s = (state[0], goal)
                a = action
                r = self.reward_func(goal, next_state[0])
                s_ = (next_state[0], goal)
                d = self.done_func(goal, next_state[0])
                self.push(s, a, r, s_, d)
