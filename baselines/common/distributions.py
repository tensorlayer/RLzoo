"""Definition of parametrized distributions. Adapted from openai/baselines"""
import numpy as np
import tensorflow as tf
from gym import spaces


class Distribution(object):
    """A particular probability distribution"""
    def set_param(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sampling from distribution. Allow explore parameters."""
        raise NotImplementedError

    def logp(self, x):
        """Calculate log probability of a sample."""
        return -self.neglogp(x)

    def neglogp(self, x):
        """Calculate negative log probability of a sample."""
        raise NotImplementedError

    def kl(self, *parameters):
        """Calculate Kullback–Leibler divergence"""
        raise NotImplementedError

    def entropy(self):
        """Calculate the entropy of distribution."""
        raise NotImplementedError


class Categorical(Distribution):
    """Creates a categorical distribution"""
    def __init__(self, ndim, logits=None):
        """
        Args:
            ndim (int): total number of actions
            logits (tensor): logits variables
        """
        self._ndim = ndim
        self._logits = logits

    def set_param(self, logits):
        """
        Args:
            logits (tensor): logits variables to set
        """
        self._logits = logits

    def sample(self):
        u = tf.random.uniform(tf.shape(self._logits), dtype=self._logits.dtype)
        return tf.argmax(self._logits - tf.math.log(-tf.math.log(u)), axis=-1)

    def logp(self, x):
        return -self.neglogp(x)

    def neglogp(self, x):
        x = tf.one_hot(x, self._ndim)
        return tf.nn.softmax_cross_entropy_with_logits(x, self._logits)

    def kl(self, logits):
        """
        Args:
            logits (tensor): logits variables of another distribution
        """
        a0 = self._logits - tf.reduce_max(self._logits, axis=-1, keepdims=True)
        a1 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(
            p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

    def entropy(self):
        a0 = self._logits - tf.reduce_max(self._logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)


class DiagGaussian(Distribution):
    """Creates a diagonal Gaussian distribution """
    def __init__(self, ndim, mean_logstd=None):
        """
        Args:
            ndim (int): the dimenstion of actions
            mean_logstd (tensor): mean and logstd stacked on the last axis
        """
        self._ndim = ndim
        self.mean = None
        self.logstd = None
        self.std = None
        if mean_logstd is not None:
            self.set_param(mean_logstd)

    def set_param(self, mean_logstd):
        """
        Args:
            mean_logstd (tensor): mean and logstd, stacked on the last axis
        """
        mean, logstd = tf.split(mean_logstd, 2, -1)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.math.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    def logp(self, x):
        return -self.neglogp(x)

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.mean) / self.std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * float(self._ndim) \
            + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, mean_logstd):
        """
        Args:
            mean_logstd (tensor): mean and logstd of another distribution,
                                  stacked on the last axis
        """
        mean, logstd = tf.split(mean_logstd, 2, -1)
        return tf.reduce_sum(
            logstd - self.logstd +
            (tf.square(self.std) + tf.square(self.mean - mean))
            / (2.0 * tf.square(tf.math.exp(logstd))) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(
            self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def make_dist(ac_space):
    """Get distribution based on action space
    Args:
        ac_space (gym.spaces.Space)
    """
    if isinstance(ac_space, spaces.Discrete):
        return Categorical(ac_space.n)
    elif isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussian(ac_space.shape[0])
    else:
        raise NotImplementedError
