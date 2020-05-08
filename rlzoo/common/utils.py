"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import os
import re

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorlayer as tl
import tensorflow as tf
from importlib import import_module


def plot(episode_rewards, algorithm_name, env_name):
    """
    plot the learning curve, saved as ./img/algorithm_name-env_name.png

    :param episode_rewards: array of floats
    :param algorithm_name: string
    :param env_name: string
    """
    path = os.path.join('.', 'img')
    name = algorithm_name + '-' + env_name
    plt.figure(figsize=(10, 5))
    plt.title(name)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, name + '.png'))
    plt.close()


def plot_save_log(episode_rewards, algorithm_name, env_name):
    """
    plot the learning curve, saved as ./img/algorithm_name-env_name.png,
    and save the rewards log as ./log/algorithm_name-env_name.npy

    :param episode_rewards: array of floats
    :param algorithm_name: string
    :param env_name: string
    """
    path = os.path.join('.', 'log')
    name = algorithm_name + '-' + env_name
    plot(episode_rewards, algorithm_name, env_name)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, name), episode_rewards)


def save_model(model, model_name, algorithm_name, env_name):
    """
    save trained neural network model

    :param model: tensorlayer.models.Model
    :param model_name: string, e.g. 'model_sac_q1'
    :param algorithm_name: string, e.g. 'SAC'
    """
    name = algorithm_name + '-' + env_name
    path = os.path.join('.', 'model', name)
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_npz(model.trainable_weights, os.path.join(path, model_name))


def load_model(model, model_name, algorithm_name, env_name):
    """
    load saved neural network model

    :param model: tensorlayer.models.Model
    :param model_name: string, e.g. 'model_sac_q1'
    :param algorithm_name: string, e.g. 'SAC'
    """
    name = algorithm_name + '-' + env_name
    path = os.path.join('.', 'model', name)
    try:
        param = tl.files.load_npz(path, model_name + '.npz')
        for p0, p1 in zip(model.trainable_weights, param):
            p0.assign(p1)
    except Exception as e:
        print('Load Model Fails!')
        raise e


def parse_all_args(parser):
    """ Parse known and unknown args """
    common_options, other_args = parser.parse_known_args()
    other_options = dict()
    index = 0
    n = len(other_args)
    float_pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    while index < n:  # only str, int and float type will be parsed
        if other_args[index].startswith('--'):
            if other_args[index].__contains__('='):
                key, value = other_args[index].split('=')
                index += 1
            else:
                key, value = other_args[index:index + 2]
                index += 2
            if re.match(float_pattern, value):
                value = float(value)
                if value.is_integer():
                    value = int(value)
            other_options[key[2:]] = value
    return common_options, other_options


def make_env(env_id):
    env = gym.make(env_id).unwrapped
    """ add env wrappers here """
    return env


def get_algorithm_module(algorithm, submodule):
    """ Get algorithm module in the corresponding folder """
    return import_module('.'.join(['rlzoo', 'algorithms', algorithm, submodule]))


def call_default_params(env, envtype, alg, default_seed=True):
    """ Get the default parameters for training from the default script """
    alg = alg.lower()
    default = import_module('.'.join(['rlzoo', 'algorithms', alg, 'default']))
    params = getattr(default, envtype)(env,
                                       default_seed)  # need manually set seed in the main script if default_seed = False
    return params


def set_seed(seed, env=None):
    """ set random seed for reproduciblity """
    if isinstance(env, list):
        assert isinstance(seed, list)
        for i in range(len(env)):
            env[i].seed(seed[i])
        seed = seed[0]  # pick one seed for np and tf
    elif env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
