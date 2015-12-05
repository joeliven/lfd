import scipy as sci
import numpy as np
import os
import config as c
import time
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import physics as sim
import sklearn as sk
from sklearn import svm
from sklearn import linear_model
import process_path as pp
import statistics as stats
import rl



def load_pweights_rweights(filename):
    pweights_list = list()
    rweights_list = list()
    with open(filename, "r") as f:
        for line in f:
            # print("line is: " + str(line))
            if line.split(",")[0].rstrip() == "pweight":
                pweights_list.append(float(line.split(",")[1].rstrip()))
            elif line.split(",")[0].rstrip() == "rweight":
                rweights_list.append(float(line.split(",")[1].rstrip()))
    pweights = np.array(pweights_list)
    rweights = np.array(rweights_list)
    print("pweights is: " + str(np.shape(pweights)))
    print(pweights)
    print("rweights is: " + str(np.shape(rweights)))
    print(rweights)
    return pweights, rweights

def load_weights_numpy(filename):
    weights = np.load(filename)
    return weights

filenames = ["SF_demo0",
                    "SF_demo1",
                    "SF_demo2",
                    "SF_demo3",
                    "SF_demo4",
                    "SF_demo5",
                    "SF_demo6",
                    "SF_demo7",
                    # "SF_demo8",
                    # "SF_demo9",
]
demos, starts, goals, modes, emos = rl.process_demos(filenames)
vnorm_bins = rl.detect_bins(demos, starts, goals)

fname="cur_opt___SF_2dp_demo_pweights_10.npy"
# fname="demo0_pweights_rweights_10.csv"
# pweights_, rweights_ = load_pweights_rweights(fname)
pweights_ = load_weights_numpy(fname)
input("checking...")
print("pweights_ is: " + str(np.shape(pweights_)))
print(pweights_)

# generalized_start = [450,67]
# generalized_goal = [86,396]

# generalized_start = [250,167]
# generalized_goal = [416,196]

# generalized_start = [150,367]
# generalized_goal = [379,112]

generalized_start = [450,467]
generalized_goal = [79,112]

# generalized_start = [250,350]
# generalized_goal = [150,225]



# pweights = rl.SARSA(rweights_, generalized_start, generalized_goal, vnorm_bins, -1)
# path, err_goal, termination_msg, path_duration = rl.create_path_from_policy(pweights, generalized_start, generalized_goal, vnorm_bins, None)

path, err_goal, termination_msg, path_duration = rl.create_path_from_policy(pweights_, generalized_start, generalized_goal, vnorm_bins, None)

paths = [path]
rl.save_paths_to_file(paths, generalized_start, generalized_goal, "test_emo", "test_mode", "aaa", "na")
# path, err_goal, termination_msg, path_duration = rl.create_path_from_policy(pweights_, start, goal, T)



