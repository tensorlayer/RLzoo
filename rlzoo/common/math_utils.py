"""
Functions for mathematics utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""


def flatten_dims(shapes):  # will be moved to common
    dim = 1
    for s in shapes:
        dim *= s
    return dim
