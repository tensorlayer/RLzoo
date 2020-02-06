"""
Functions for mathematics utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import operator
import os
import re
import random
import gym


import matplotlib.pyplot as plt
from importlib import import_module
import numpy as np

import tensorlayer as tl


def flatten_dims(shapes):  # will be moved to common
    dim = 1
    for s in shapes:
        dim *= s
    return dim