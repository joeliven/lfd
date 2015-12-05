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

subject = "KEENAN"
emos = ["LOOP",
            "ANGER",
            "SKZ",
            "CONTROL",
]

demo_filenames = [["KEENAN_LOOP_demo0",
                    "KEENAN_LOOP_demo1",
                    "KEENAN_LOOP_demo2",
                    "KEENAN_LOOP_demo3",
                    "KEENAN_LOOP_demo4",
                    "KEENAN_LOOP_demo5",
                    "KEENAN_LOOP_demo6",
                    "KEENAN_LOOP_demo7",
                    "KEENAN_LOOP_demo8",
                    "KEENAN_LOOP_demo9",
                    ],
                ["KEENAN_ANGER_demo0",
                    "KEENAN_ANGER_demo1",
                    "KEENAN_ANGER_demo2",
                    "KEENAN_ANGER_demo3",
                    "KEENAN_ANGER_demo4",
                    "KEENAN_ANGER_demo5",
                    "KEENAN_ANGER_demo6",
                    "KEENAN_ANGER_demo7",
                    ],
                ["KEENAN_SKZ_demo0",
                    "KEENAN_SKZ_demo1",
                    "KEENAN_SKZ_demo2",
                    "KEENAN_SKZ_demo3",
                    "KEENAN_SKZ_demo4",
                    "KEENAN_SKZ_demo5",
                    ],
                    ["CONTROL_demo0",
                    "CONTROL_demo1",
                    "CONTROL_demo2",
                    "CONTROL_demo3",
                    ],
]

pweight_filenames = ["_opt___KEENAN_LOOP_demo_pweights_10.npy",
    "_opt___KEENAN_ANGER_demo_pweights_10.npy",
    "_opt___KEENAN_SKZ_demo_pweights_10.npy",
    "_opt___CONTROL_demo_pweights_10.npy",
]

with open ("exp" + str(subject) + "_truth.txt", "w") as truth_file:
    truth_file.write("subject:" + str(subject))
    truth_file.write("\n")
    truth_file.write("emos:")
    for emo in range(0, len(emos)):
        truth_file.write(str(emos[emo]) + ",")
    truth_file.write("\n")
    truth_file.write("******************************************************************")
    truth_file.write("\n")
    truth_file.write("TRUTH VALUES")
    truth_file.write("\n")

    path_filenames = list()
    for i in range(0, 50):
        print("i: " + str(i))
        rand_nums = np.random.random_integers(0,500,4)
        start =  np.array([rand_nums[0], rand_nums[1]])
        goal =  np.array([rand_nums[2], rand_nums[3]])
        # print("start is: " + str(start))
        # print("goal is: " + str(goal))
        # input("above while")
        j = 1
        start_goal_dist = np.linalg.norm(start - goal)
        while(start_goal_dist < 150):
            # input("in while")

            print("j is: " + str(j))
            j+=1
            rand_nums = np.random.random_integers(0,500,4)
            start =  np.array([rand_nums[0], rand_nums[1]])
            goal =  np.array([rand_nums[2], rand_nums[3]])
            start_goal_dist = np.linalg.norm(start - goal)
        # input("after while")


        emo = np.random.random_integers(0,len(emos)-1)
        # print("emo is: "  + str(emos[emo]))
        print("demo_filenames is: " + str(demo_filenames[emo]))
        demos, starts, goals, modes, emos_ignore = rl.process_demos(demo_filenames[emo])
        print("demos is:...")
        vnorm_bins = rl.detect_bins(demos, starts, goals)
        pweights_ = load_weights_numpy(pweight_filenames[emo])
        print("pweights_ is: " + str(pweights_))
        path, err_goal, termination_msg, path_duration = rl.create_path_from_policy(pweights_, start, goal, vnorm_bins, None)
        paths = [path]
        print("path is: "+ str(path))
        path_filename = "exp" + str(subject) + "_"
        rl.save_paths_to_file(paths, start, goal, emos[emo], "subject=" + str(subject), path_filename, i)
        path_filenames.append(path_filename + str(i) + ".csv")
        truth_file.write(str(i) + ":" + str(emos[emo]))
        truth_file.write("\n")
        i+=1

with open ("exp" + str(subject) + "_pathfile_names.txt", "w") as f:
    for path_filename in path_filenames:
        f.write(str(path_filename))
        f.write("\n")
        # input("checking...")
        # print("pweights_ is: " + str(np.shape(pweights_)))
        # print(pweights_)


###################################################################
###################################################################
###################################################################
###################################################################
# generalized_start = [450,67]
# generalized_goal = [86,396]

# generalized_start = [250,167]
# generalized_goal = [416,196]

# generalized_start = [150,367]
# generalized_goal = [379,112]

# generalized_start = [450,467]
# generalized_goal = [79,112]

# generalized_start = [250,350]
# generalized_goal = [150,225]

# pweights = rl.SARSA(rweights_, generalized_start, generalized_goal, vnorm_bins, -1)
# path, err_goal, termination_msg, path_duration = rl.create_path_from_policy(pweights, generalized_start, generalized_goal, vnorm_bins, None)

# path, err_goal, termination_msg, path_duration = rl.create_path_from_policy(pweights_, start, goal, T)
