import scipy as sci
import numpy as np
import os
import config as c
import time
import matplotlib.pyplot as plt
from scipy import interpolate
import math

# STATE:
    # x
    # y
    # x_dot
    # y_dot

# index locations in state array:
_x = 0
_y = 1
_xd = 2
_yd = 3
# _xdd = 4
# _ydd = 5

#non-deterministic actions to add some stochasticity to the model...see if it improves learning???
def get_next_state(cur_state, action):
    action_x = action[0]
    action_y = action[1]
    u = np.random.uniform(0,1)
    if u < 0.:
    # if u < .01:
        ux = np.random.uniform(0,1)
        if ux < .5:
            action_x -= 1.
        else:
            action_x += 1.
        uy = np.random.uniform(0,1)
        if uy < .5:
            action_y -= 1.
        else:
            action_y += 1.

    next_state = np.zeros_like(cur_state)

    next_state[_x] = cur_state[_x] + cur_state[_xd]
    next_state[_xd] = cur_state[_xd] + action_x

    next_state[_y] = cur_state[_y] + cur_state[_yd]
    next_state[_yd] = cur_state[_yd] + action_y

    return next_state

# def get_next_state(cur_state, action):
#     next_state = np.zeros_like(cur_state)

#     next_state[_x] = cur_state[_x] + cur_state[_xd]
#     next_state[_xd] = cur_state[_xd] + action[0]

#     next_state[_y] = cur_state[_y] + cur_state[_yd]
#     next_state[_yd] = cur_state[_yd] + action[1]

#     return next_state




############################################################
############################################################
############################################################
    # next_state[_xdd] = cur_state[_xdd] + action[0]
    # next_state[_ydd] = cur_state[_ydd] + action[1]


