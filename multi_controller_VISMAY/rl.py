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


##############################################################
#                  TEMP
##############################################################
plog_global = False
flog_global = False

demo_group = "___VISMAY3_demo"
filenames = ["VISMAY3_demo0",
                    "VISMAY3_demo1",
                    "VISMAY3_demo2",
                    "VISMAY3_demo3",
                    "VISMAY3_demo4",
                    "VISMAY3_demo5",
                    # "CONTROL_demo6",
                    # "CONTROL_demo7",
                    # "CONTROL_demo8",
                    # "CONTROL_demo9",
]
save_recreates = "___VISMAY3_"
save_recreates_cur_opt = "___VISMAY3_opt_"
logfile_name = "logfile" + str(demo_group) + ".txt"

##############################################################
#                  PARAMETERS and CONSTANTS
##############################################################
discount = 0.9
on_policy = False
epsilon_IRL = 0.0001 # arbitrary right now
epsilon_RL = .05 # no idea on what to set this to
epsilon_RL_policy = .05
# epsilon_RL_convergence = .01
epsilon_RL_convergence = .001
# epsilon_RL_convergence = .1
# lr_base = .5
lr_base = .75
lmbda = .1

use_svm = True
dparts = 4

max_iters = 21
max_iters_per_episode = 1500
# max_iters_per_episode = 2000
num_rollouts = 25

num_vnorm_bins = 5

##############################################################
#                  DEFINING FEATURES and other PRELIMINARIES
##############################################################
# position, velocity, and acceleration mins and maxs...predefined for normalization purposes:
x_min = 0.
y_min = 0.
x_max = 500.
y_max = 500.

tparts = 2

demo_state_dim = 6
state_dim = 4
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

# ACTIONS:
    # set xdd # limited to values in "global_action_list"
    # set ydd # limited to values in "global_action_list"
# index locations in action array:
_xdd = 0
_ydd = 1
# global_action_list = np.array([-1,-2,-1,0,1,2,1])
global_action_list = np.array([-2,-1,0,1,2])
num_actions = len(global_action_list)

global_actions = np.zeros((num_actions**2,2))
_idx_ = 0
for x in global_action_list:
    for y in global_action_list:
        global_actions[_idx_,_x] = float(x)
        global_actions[_idx_,_y] = float(y)
        _idx_ += 1

if plog_global:
    print("global_actions is: " + str(np.shape(global_actions)))
    print(global_actions)
    # input("global_actions")

index = 0
#####FEATURES for reward learning (and corresponding index locations in the features array) #####

### from STATE ###
## temporal distance feature (1):
k_dp = 0
# dp_in = index        # is this x,y state in the current distance partition? True/False
# index += 1
## absolute value of velocity features (5):
k_velocity = 5
v0 = index        # math.sqrt(xd**2 + yd**2) == vnorm_bins_0
index += 1
v1 = index        # math.sqrt(xd**2 + yd**2) == vnorm_bins_1
index += 1
v2 = index        # math.sqrt(xd**2 + yd**2) == vnorm_bins_2
index += 1
v3 = index        # math.sqrt(xd**2 + yd**2) == vnorm_bins_3
index += 1
v4 = index        # math.sqrt(xd**2 + yd**2) == vnorm_bins_4
index += 1

### from ACTION ###
## acceleration and velocity interaction features (25):
k_acceleration = 0
# v0_ss = index    # the result of an action causing the norm of the velocity to decrease by 2 units or more
# index += 1
# v0_s = index    # the result of an action causing the norm of the velocity to decrease by 1 unit or more but less than 2 units
# index += 1
# v0_ = index    # the result of an action causing the norm of the velocity to change by less than 1 unit in either direction
# index += 1
# v0_f = index    # the result of an action causing the norm of the velocity to increase by 1 unit or more but less than 2 units
# index += 1
# v0_ff = index    # the result of an action causing the norm of the velocity to increase by 2 units or more
# index += 1
# v1_ss = index    # the result of an action causing the norm of the velocity to decrease by 2 units or more
# index += 1
# v1_s = index    # the result of an action causing the norm of the velocity to decrease by 1 unit or more but less than 2 units
# index += 1
# v1_ = index    # the result of an action causing the norm of the velocity to change by less than 1 unit in either direction
# index += 1
# v1_f = index    # the result of an action causing the norm of the velocity to increase by 1 unit or more but less than 2 units
# index += 1
# v1_ff = index    # the result of an action causing the norm of the velocity to increase by 2 units or more
# index += 1
# v2_ss = index    # the result of an action causing the norm of the velocity to decrease by 2 units or more
# index += 1
# v2_s = index    # the result of an action causing the norm of the velocity to decrease by 1 unit or more but less than 2 units
# index += 1
# v2_ = index    # the result of an action causing the norm of the velocity to change by less than 1 unit in either direction
# index += 1
# v2_f = index    # the result of an action causing the norm of the velocity to increase by 1 unit or more but less than 2 units
# index += 1
# v2_ff = index    # the result of an action causing the norm of the velocity to increase by 2 units or more
# index += 1
# v3_ss = index    # the result of an action causing the norm of the velocity to decrease by 2 units or more
# index += 1
# v3_s = index    # the result of an action causing the norm of the velocity to decrease by 1 unit or more but less than 2 units
# index += 1
# v3_ = index    # the result of an action causing the norm of the velocity to change by less than 1 unit in either direction
# index += 1
# v3_f = index    # the result of an action causing the norm of the velocity to increase by 1 unit or more but less than 2 units
# index += 1
# v3_ff = index    # the result of an action causing the norm of the velocity to increase by 2 units or more
# index += 1
# v4_ss = index    # the result of an action causing the norm of the velocity to decrease by 2 units or more
# index += 1
# v4_s = index    # the result of an action causing the norm of the velocity to decrease by 1 unit or more but less than 2 units
# index += 1
# v4_ = index    # the result of an action causing the norm of the velocity to change by less than 1 unit in either direction
# index += 1
# v4_f = index    # the result of an action causing the norm of the velocity to increase by 1 unit or more but less than 2 units
# index += 1
# v4_ff = index    # the result of an action causing the norm of the velocity to increase by 2 units or more
# index += 1
# ## angular motion style features (6):
k_angular_motion_style = 6
v_ang_0 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 0 and < 30
index += 1
v_ang_1 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 30 and < 60
index += 1
v_ang_2 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 60 and < 90
index += 1
v_ang_3 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 90 and < 120
index += 1
v_ang_4 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 120 and < 150
index += 1
v_ang_5 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 150 and < 180
index += 1
# ## angular motion relative to goal features (6):
k_angular_motion_rel_goal = 6
g_ang_0 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 0 and < 30
index += 1
g_ang_1 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 30 and < 60
index += 1
g_ang_2 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 60 and < 90
index += 1
g_ang_3 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 90 and < 120
index += 1
g_ang_4 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 120 and < 150
index += 1
g_ang_5 = index     # v_ang_0 = True if (180/pi) * math.acos(np.dot([v1,v2])/(np.linalg.norm(v1)*np.linalg.norm(v2))) >= 150 and < 180
index += 1
## location features relative to goal (1):
k_loc_goal = 2
# closer_x = index    # closer_to_goal_x = True/False
# index += 1
# closer_y = index    # closer_to_goal_y = True/False
# index += 1
closer_euc = index # closer_to_goal_euc = True/False
index += 1
in_goal = index # closer_to_goal_euc = True/False
index += 1
## location features relative to grid world (1):
k_loc_grid = 1
# k_loc_grid = 3
ob0 = index # out_of_bounds = True/False
index += 1
# ob1 = index # out_of_bounds = True/False
# index += 1
# ob2 = index # out_of_bounds = True/False
# index += 1
# ob3 = index # out_of_bounds = True/False
# index += 1
# ob4 = index # out_of_bounds = True/False
# index += 1

# number of features:
full_k = k_dp + k_velocity + k_acceleration + k_loc_goal + k_loc_grid + k_angular_motion_style + k_angular_motion_rel_goal
k = full_k

##############################################################
#                  PROCESSING
##############################################################
def process_demos(filenames):
    plog = False
    flog = False
    demos = list()
    starts = list()
    goals = list()
    modes = list()
    emos = list()
    for fname in filenames:
        demo, start, goal, mode, emo = pp.execute(fname, False)
        demos.append(demo)
        starts.append(start)
        goals.append(goal)
        modes.append(mode)
        emos.append(emo)
    if plog:
        print("demos is: (in process_demos() ) ")
        print(demos)
    return demos, starts, goals, modes, emos

def get_demos(filenames):
    demos = list()
    for fname in filenames:
        with open(fname, "r") as f:
            demo = list()
            for line in f:
                temp_list = [float(x) for x in line.split(",") if x != "\n"]
                demo.append(temp_list)
            demo = np.array(demo)
            demos.append(demo)
    return demos

def save_pweights_rweights_to_file(save_name, demo_group, ct, err_goal, path_duration, termination_msg, pweights, rweights):
    save_name += ".csv"
    with open(save_name, "w") as f:
        f.write("demogroup:" + str(demo_group))
        f.write("\n")
        f.write("ct:" + str(ct))
        f.write("\n")
        f.write("err_goal:" + str(err_goal))
        f.write("\n")
        f.write("path_duration:" + str(path_duration))
        f.write("\n")
        f.write("termination_msg:" + str(termination_msg))
        f.write("\n")
        if pweights != None:
            f.write("pweights:")
            f.write("\n")
            for weight in pweights:
                f.write("pweight," + str(weight))
                f.write("\n")
        f.write("rweights:")
        f.write("\n")
        for weight in rweights:
            f.write("rweight," + str(weight))
            f.write("\n")
        f.write("end")

def save_weights_to_file_numpy(save_name, weights):
    np.save(save_name, weights)

def save_paths_to_file(paths, start, goal, emo, mode, save_name, IRL_iter):
    start_ = [start[0]-5, start[1]-5, start[0]+5, start[1]+5, ]
    goal_ = [goal[0]-5, goal[1]-5, goal[0]+5, goal[1]+5, ]
    ct = IRL_iter
    for path in paths:
        filename = str(save_name) + str(ct) + ".csv"
        with open(filename, "w") as f:
            f.write("mode:"+str(mode))
            f.write("\n")
            f.write("emotion:"+str(emo))
            f.write("\n")
            f.write("start:")
            for i in range(0,len(start_)):
                f.write(str(start_[i]))
                if i < len(start_) - 1:
                    f.write(",")
            f.write("\n")
            f.write("goal:")
            for i in range(0,len(goal_)):
                f.write(str(goal_[i]))
                if i < len(goal_) - 1:
                    f.write(",")
            f.write("\n")
            for point in path:
                f.write(str(point[0]))
                f.write(",")
                f.write(str(point[1]))
                f.write("\n")
        # ct += 1

def detect_bins(demos, starts, goals):
    plog = False
    flog = False
    vnorms = list()
    for dp in range(0,dparts):
        vnorms.append(list())
    for dem in range (0, len(demos)):
        demo = demos[dem]
        start = starts[dem]
        goal = goals[dem]
        dist_tot = math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)
        for t in range(0, len(demo)):
            dist_cur = math.sqrt((goal[0]-demo[t,_x])**2 + (goal[1]-demo[t,_y])**2)
            dist_ratio = float(dist_cur) / float(dist_tot)
            dist_ratio = 1. - dist_ratio
            if dist_ratio < 0.:
                dist_ratio = 0.
            cur_dp  = -1
            for dp in range(0, dparts):
                lower_bound_ratio = float(dp) / float(dparts)
                upper_bound_ratio = float(dp+1) / float(dparts)
                if plog:
                    print("lower_bound_ratio is: " + str(lower_bound_ratio))
                    print("upper_bound_ratio is: " + str(upper_bound_ratio))
                if dist_ratio >= lower_bound_ratio and dist_ratio < upper_bound_ratio:
                    cur_dp = dp

            v = [demo[t,_xd],demo[t,_yd]]
            vnorm = np.linalg.norm(v)
            vnorms[cur_dp].append(vnorm)

    vnorm_bins = np.zeros((dparts, num_vnorm_bins, 2))
    for dp in range(0, dparts):
        vnorm_mean = stats.mean(vnorms[dp])
        vnorm_stdev = stats.stdev(vnorms[dp], vnorm_mean)

        last_split = vnorm_mean + 2.*vnorm_stdev
        bin_width = last_split / float(num_vnorm_bins - 1)
        vnorm_splits = list()
        for i in range(0, num_vnorm_bins):
            vnorm_splits.append(0 + i*bin_width)
        for i in range(0, num_vnorm_bins):
            if i < num_vnorm_bins-1:
                vnorm_bins[dp,i,0] = vnorm_splits[i]
                vnorm_bins[dp,i,1] = vnorm_splits[i+1]
            else:
                vnorm_bins[dp,i,0] = vnorm_splits[i]
                vnorm_bins[dp,i,1] = np.inf

    print("vnorm_bins is: " + str(np.shape(vnorm_bins)))
    print(vnorm_bins)
    # input("check and make sure that vnorm_bins looks right...")
    return vnorm_bins


##############################################################
#                  RL SOLVER (continuous)
##############################################################
def SARSA(rweights, start, goal, vnorm_bins, IRL_iter):
    # NOTES:
        # s = "state"
        # s_ = "next state"
        # a = "action"
        # a_ = "next action"
        # Fsa = "features mapping of state, s, and action, a"
        # Fsa_ = "features mapping of state, s_, and action, a_"

    # randomly initialize the weights that parameterize the value function approximation:
    epsilon = epsilon_RL_policy
    # lr = lr_start
    # pweights = np.random.uniform(-1.,1.,k+1)
    pweights = np.random.uniform(-1.,1.,k+1)
    pweights_norm = np.linalg.norm(pweights)
    pweights = pweights/pweights_norm
    # pweights_norm = np.linalg.norm(pweights)

    # print("norm of random pweights is: " + str(pweights_norm))
    # print("pweights is: ")
    # print(pweights)

    dif = np.inf
    ct = 0
    episode = 0
    convergence = 0.001
    if IRL_iter < 5:
        convergence = 0.01
    elif IRL_iter >= 5 and IRL_iter< 15:
        convergence = 0.01
    elif IRL_iter >= 15 and IRL_iter< 25:
        convergence = 0.005
    elif IRL_iter >= 25 and IRL_iter< 35:
        convergence = 0.001

    while dif > convergence: # loop through multiple episodes until convergence
        _pweights = pweights
        trace = np.zeros(k+1)
        s = np.zeros(6)
        # s[_x] = start[0]
        # s[_y] = start[1]
        s[_x] = np.random.random_integers(x_min, x_max)
        s[_y] = np.random.random_integers(y_min, y_max)
        s[_xd] = 0. #DESQ: should xdot and ydot be initialized to 0. or no? Maybe a small random value? Or 1 in direction of goal?
        s[_yd] = 0. #DESQ: should xdot and ydot be initialized to 0. or no? Maybe a small random value? Or 1 in direction of goal?
        a, idx_a, Qsa, Fsa =  select_action_from_policy(pweights, s, epsilon, start, goal, vnorm_bins)
        # a = global_actions[idx]
        # Fsa = np.zeros(k+1)
        # Fsa[0:k] = map_features(s, start, goal, a)
        # Fsa[k] = 1.
        # Qsa = np.dot(pweights, Fsa)

        termination = False # iterate through each step of a given episode
        t = 0
        while not termination and t < max_iters_per_episode:
            lr = lr_base**((math.log10(ct + 10))**2)
            # trace += Fsa
            trace = Fsa
            r = np.dot(rweights,Fsa[0:-1])
            # r = np.dot(rweights,Fsa)
            delta = r - Qsa

            s_ = sim.get_next_state(s, a)
            a_, idx_a_, Qsa_, Fsa_ = select_action_from_policy(pweights, s_, epsilon, start, goal, vnorm_bins)
            delta += discount*Qsa_
            # print("r is: " + str(r))
            # print("Qsa is: " + str(Qsa))
            # print("Qsa_ is: " + str(Qsa_))
            # print("delta is: " + str(delta))
            # input("analyze...")

            pweights = pweights + (lr*delta*trace)
            pweights_norm = np.linalg.norm(pweights)
            pweights =  pweights / pweights_norm
            # trace += (discount*lmbda*trace)
            ct += 1
            s = s_
            a = a_
            Qsa = Qsa_
            Fsa = Fsa_
            # check state for termination conditions:
            termination, termination_result, termination_reason, termination_details = is_state_terminal(s, goal)
            if termination:
                termination_msg = "IRL_iter: " + str(IRL_iter) + "\tCt: " + str(ct) + "\tEpisode: " + str(episode) + "\tIteration: " + str(t) + "\tlr=" + str(lr)
                termination_msg += "\tResult: " + str(termination_result) + "\tReason: " + str(termination_reason)
                termination_msg += "\nState: " + str(termination_details)
            t += 1
            ct += 1
        episode += 1

        dif = np.linalg.norm(_pweights - pweights)
        print("\n\n***********************************************************************")
        print(termination_msg)
        # input("pausing:")
        print("pweights is: ")
        print(pweights)
        print("dif is: " + str(dif))
        # input("pausing after completion of an episode::")
        _pweights = pweights
    print("dif is: " + str(dif))
    print("pweights is: ")
    print(pweights)
    # input("pausing:")

    return pweights


#STANDARD WAY OF DOING IT...
def optimal_action(pweights, state, start, goal, vnorm_bins):
    # plog = True
    plog = False
    flog = False
    Qsa_array = np.zeros(num_actions**2)
    Fsa_list = list()
    idx = 0
    if plog:
        pass
        # print("pweights (policy) is: ")
        # print(pweights)
        # input("in optimal_action():")
    for action in global_actions:
        Fsa = np.zeros(k+1)
        Fsa[0:k], dp = map_features(state, start, goal, vnorm_bins, action)
        Fsa[k] = 1.
        Fsa_list.append(Fsa)
        Qsa_array[idx] = np.dot(pweights[dp], Fsa)
        idx += 1
    optimal_idx = -1
    optimal_idxs = np.argwhere(Qsa_array == np.amax(Qsa_array))
    optimal_idxs = optimal_idxs.flatten()
    # print("optimal_idxs is: " + str(optimal_idxs))
    if len(optimal_idxs) == 1:
        optimal_idx = optimal_idxs[0]
    else:
        optimal_idx = optimal_idxs[np.random.random_integers(0, len(optimal_idxs)-1)]

        # angles = np.zeros_like(optimal_idxs)
        # i = 0
        # for idx in optimal_idxs:
        #     a = global_actions[idx]
        #     s_ = sim.get_next_state(state, a)
        #     s__ = sim.get_next_state(s_, [0,0])
        #     v_goal = [float(goal[0]-s_[0]), float(goal[1]-s_[1])]
        #     v_action = [float(s__[_x]-s_[_x]), float(s__[_y]-s_[_y])]

        #     dot_prod = np.dot(v_goal,v_action)
        #     vnorm_goal = np.linalg.norm(v_goal)
        #     vnorm_action = np.linalg.norm(v_action)
        #     if (vnorm_goal * vnorm_action != 0):
        #         ratio = dot_prod / (vnorm_goal * vnorm_action)
        #         if ratio > 1.:
        #             ratio = 1.
        #         elif ratio < -1.:
        #             ratio = -1.
        #         angle = (180./np.pi) * math.acos(ratio)
        #     else:
        #         angle = 1000.
        #     angles[i] = angle
        #     i += 1
        # opt = np.argwhere(angles == np.amin(angles))
        # opt = opt.flatten()
        # if len(opt == 1):
        #     optimal_idx = optimal_idxs[opt[0]]
        # else:
        #     optimal_idx = optimal_idxs[opt[np.random.random_integers(0, len(opt)-1)]]

    optimal_a = global_actions[optimal_idx]
    if plog:
        print("optimal_idx is: " + str(optimal_idx))
        print("optimal_a is: " + str(optimal_a))
    Qsa = Qsa_array[optimal_idx]
    Fsa = Fsa_list[optimal_idx]
    return optimal_a, optimal_idx, Qsa, Fsa, dp

# with lookahead
def optimal_action_wlookahead(pweights, state, start, goal, vnorm_bins):
    # plog = True
    plog = False
    flog = False
    Qsa_array = np.zeros(num_actions**2)
    Fsa_list = list()
    idx = 0
    if plog:
        pass
        # print("pweights (policy) is: ")
        # print(pweights)
        # input("in optimal_action():")
    for action in global_actions:
        Fsa = np.zeros(k+1)
        Fsa[0:k], dp = map_features(state, start, goal, vnorm_bins, action)
        Fsa[k] = 1.
        Fsa_list.append(Fsa)


        qsa_cur = np.dot(pweights, Fsa)
        Qsa_array_ = np.zeros(num_actions**2)
        qsa_next = 0.
        idx_ = 0
        for action_ in global_actions:
            state_ = sim.get_next_state(state, action_)
            Fsa_ = np.zeros(k+1)
            Fsa_[0:k], dp_ = map_features(state_, start, goal, vnorm_bins, action_)
            Fsa_[k] = 1.
            # Fsa_list.append(Fsa_)
            Qsa_array_[idx_] = np.dot(pweights, Fsa_)
            idx_ += 1
        qsa_next = np.amax(Qsa_array_)
        Qsa_array[idx] = qsa_cur + discount*qsa_next

        idx += 1
    optimal_idx = -1
    optimal_idxs = np.argwhere(Qsa_array == np.amax(Qsa_array))
    optimal_idxs = optimal_idxs.flatten()
    # print("optimal_idxs is: " + str(optimal_idxs))
    if len(optimal_idxs) == 1:
        optimal_idx = optimal_idxs[0]
    else:
        optimal_idx = optimal_idxs[np.random.random_integers(0, len(optimal_idxs)-1)]
        # angles = np.zeros_like(optimal_idxs)
        # i = 0
        # for idx in optimal_idxs:
        #     a = global_actions[idx]
        #     s_ = sim.get_next_state(state, a)
        #     s__ = sim.get_next_state(s_, [0,0])
        #     v_goal = [float(goal[0]-s_[0]), float(goal[1]-s_[1])]
        #     v_action = [float(s__[_x]-s_[_x]), float(s__[_y]-s_[_y])]

        #     dot_prod = np.dot(v_goal,v_action)
        #     vnorm_goal = np.linalg.norm(v_goal)
        #     vnorm_action = np.linalg.norm(v_action)
        #     if (vnorm_goal * vnorm_action != 0):
        #         ratio = dot_prod / (vnorm_goal * vnorm_action)
        #         if ratio > 1.:
        #             ratio = 1.
        #         elif ratio < -1.:
        #             ratio = -1.
        #         angle = (180./np.pi) * math.acos(ratio)
        #     else:
        #         angle = 1000.
        #     angles[i] = angle
        #     i += 1
        # opt = np.argwhere(angles == np.amin(angles))
        # opt = opt.flatten()
        # if len(opt == 1):
        #     optimal_idx = optimal_idxs[opt[0]]
        # else:
        #     optimal_idx = optimal_idxs[opt[np.random.random_integers(0, len(opt)-1)]]


    # optimal_idx = np.argmax(Qsa_array)
    optimal_a = global_actions[optimal_idx]
    if plog:
        print("optimal_idx is: " + str(optimal_idx))
        print("optimal_a is: " + str(optimal_a))
    Qsa = Qsa_array[optimal_idx]
    Fsa = Fsa_list[optimal_idx]
    return optimal_a, optimal_idx, Qsa, Fsa


def select_action_from_policy(pweights, state, epsilon, start, goal, vnorm_bins):
    u = np.random.uniform(0.,1.)
    if u < (1. - epsilon):
        # print("optimal action being taken")
        optimal_a, idx, Qsa, Fsa, dp = optimal_action(pweights, state, start, goal, vnorm_bins)
        return optimal_a, idx, Qsa, Fsa, dp
    else:
        # print("RANDOM action being taken")
        random_idx = np.random.random_integers(0, (num_actions**2) -1)
        random_a = global_actions[random_idx]
        Fsa = np.zeros(k+1)
        Fsa[0:k], dp = map_features(state, start, goal, vnorm_bins, random_a)
        Fsa[k] = 1.
        Qsa = np.dot(pweights, Fsa)
        return random_a, random_idx,  Qsa, Fsa, dp


def is_state_terminal(s, goal):
    termination = False
    buf = 50
    termination_result = ""
    termination_reason = ""
    termination_details = ""
    if (s[_x] >= goal[_x]-8) and (s[_x] <= goal[_x]+8) and (s[_y] >= goal[_y]-8) and (s[_y] <= goal[_y]+8):
        termination = True
        termination_result = "SUCCESS!!!"
        termination_reason = "in goal"
    elif (s[_x] < x_min - buf):
        termination = True
        termination_result = "failure"
        termination_reason = "x too small"
    elif (s[_x] > x_max + buf):
        termination = True
        termination_result = "failure"
        termination_reason = "x too big"
    elif (s[_y] < y_min - buf):
        termination = True
        termination_result = "failure"
        termination_reason = "y too small"
    elif (s[_y] > y_max + buf):
        termination = True
        termination_result = "failure"
        termination_reason = "y too big"
    if termination:
        termination_details = "x=" + str(s[_x]) + ", y=" + str(s[_y]) + ", xd=" + str(s[_xd])  + ", yd=" + str(s[_yd])
    return termination, termination_result, termination_reason, termination_details


def create_path_from_policy(pweights, start, goal, vnorm_bins, T):
    # plog = True
    plog = False
    flog = False

    path = list()
    s = np.zeros(state_dim)
    s[_x] = start[0]
    s[_y] = start[1]
    s[_xd] = 0
    s[_yd] = 0

    t = 0
    termination = False
    while not termination and t < max_iters_per_episode:
    # while not termination:
        # print("t is: " + str(t))
        path.append([s[_x],s[_y]])
        a, idx_a, Qsa, Fsa, dp =  select_action_from_policy(pweights, s, 0., start, goal, vnorm_bins)
        s = sim.get_next_state(s, a)
        termination, termination_result, termination_reason, termination_details = is_state_terminal(s, goal)
        t += 1
    termination_msg = "\tResult: " + str(termination_result) + "\tReason: " + str(termination_reason)
    termination_msg += "\nState: " + str(termination_details)
    path = np.array(path)
    if plog:
        print("path is: " + str(np.shape(path)))
        print(path)
    err_goal = math.sqrt((path[-1,0]-goal[0])**2 + (path[-1,1]-goal[1])**2)
    return path, err_goal, termination_msg, t



##############################################################
#                  INVERSE RL (continuous)
##############################################################


#############################################################3
#############################################################3
def optimize_reward_weights_svm(Efeats, Efeats_e): #TODO: check and make sure this works now that I've separated Efeats and Efeats_e
    # print("np.shape(Efeats) is: " + str(np.shape(Efeats)))
    # print("np.shape(Efeats_e) is: " +str(np.shape(Efeats_e)))
    training_vectors = np.zeros((len(Efeats)+1,len(Efeats[0])))
    # print("np.shape(training_vectors) is: " +str(np.shape(training_vectors)))
    training_vectors[0,:] = Efeats_e
    for i in range(1, len(training_vectors)):
        training_vectors[i,:] = np.array(Efeats[i-1])
    # print("training_vectors is: ")
    # print(training_vectors)

    # training_vectors = np.array(Efeats)
    labels = np.zeros(len(training_vectors), np.int)
    labels[0] = 1 # set the label for the experts feature expectations to 1...all other labels are set to 0
    clf = sk.svm.SVC(kernel='linear')
    # clf.fit(training_vectors, labels)
    clf.fit(training_vectors, np.reshape(labels, (-1,1)))
    w = clf.coef_[0]
    print("w is: " + str(w))
    norm = np.linalg.norm(w)
    print("norm is: " + str(norm))
    rweights = w / norm
    norm_post = np.linalg.norm(rweights)
    print("norm_post is: " + str(norm_post))
    print("shape of rweights is: " + str(np.shape(rweights)))
    print("rweights is: " + str(rweights))
    margin = math.fabs((clf.decision_function(clf.support_vectors_[0]))[0])
    return margin, rweights

def optimize_reward_weights_projection(Efeats, Efeats_bar, Efeats_e): # QQQ
    margin = np.inf
    Efeats_e = np.array(Efeats_e)
    Efeats = np.array(Efeats)
    Efeats_bar = np.array(Efeats_bar)
    rweights = np.zeros(k)
    if iteration == 1:
        rweights = Efeats_e - Efeats[0]
        margin = np.linalg.norm(rweights)
    else:
        numerator = np.dot((Efeats[i-1]-Efeats_bar[i-2]),(Efeats_e - Efeats_bar[i-2]))
        denominator = np.dot((Efeats[i-1]-Efeats_bar[i-2]),(Efeats[i-1] - Efeats_bar[i-2]))
        ratio = numerator/denominator
        additive = ratio*(Efeats[i-1] - Efeats_bar[i-2])

        Efeats_bar_iprev = (Efeats_bar[i-2] + additive)
        rweights = Efeats_e - Efeats_bar_iprev
        margin = np.linalg.norm(rweights)
    return margin, rweights, Efeats_bar_iprev


#############################################################3
#############################################################3

def map_features(state, start, goal, vnorm_bins_all, action=None):
    k = full_k
    plog = False
    flog = False
    feats = np.zeros(k, np.int)

    state_next = sim.get_next_state(state, action)
    cur_dist_goal_x = math.fabs(goal[0] - state[_x])
    cur_dist_goal_y = math.fabs(goal[1] - state[_y])
    cur_dist_goal_euc = math.sqrt((cur_dist_goal_x**2) + (cur_dist_goal_y**2))

    state_next_dist_goal_x = math.fabs(goal[0] - state_next[_x])
    state_next_dist_goal_y = math.fabs(goal[1] - state_next[_y])
    state_next_dist_goal_euc = math.sqrt((state_next_dist_goal_x**2) + (state_next_dist_goal_y**2))

    # we don't care about the action here bc all we need is the x,y position from the projection of two states out in time:
    state_next_next = sim.get_next_state(state_next, [0,0])
    resulting_dist_goal_x = math.fabs(goal[0] - state_next_next[_x])
    resulting_dist_goal_y = math.fabs(goal[1] - state_next_next[_y])
    resulting_dist_goal_euc = math.sqrt((resulting_dist_goal_x**2) + (resulting_dist_goal_y**2))
    if plog:
        print("goal[x] - cur_state[_x] =  " + str(goal[0]) + " - "+ str(state[_x]))
        print("goal[x] - state_next_next[_x] =  " + str(goal[0]) + " - "+ str(state_next_next[_x]))
        print("dif_x = " + str(cur_dist_goal_x - resulting_dist_goal_x))
        print("goal[y] - cur_state[_y] =  " + str(goal[1]) + " - "+ str(state[_y]))
        print("goal[y] - state_next_next[_y] =  " + str(goal[1]) + " - "+ str(state_next_next[_y]))
        print("dif_y = " + str(cur_dist_goal_y - resulting_dist_goal_y))
        print("dif_euc = " + str(cur_dist_goal_euc - resulting_dist_goal_euc))

    d_tot_euc = math.sqrt((start[_x] - goal[_x])**2 + (start[_y] - goal[_y])**2) # total euclidean distance from the start to the goal
    cur_d_ratio = float(cur_dist_goal_euc) / float(d_tot_euc)
    cur_d_ratio = 1.0 - cur_d_ratio
    if cur_d_ratio < 0.:
        cur_d_ratio = 0.
    cur_dp  = -1
    if plog:
        print("cur_dist_goal_euc is: " + str(cur_dist_goal_euc))
        print("d_tot_euc is: " + str(d_tot_euc))
        print("cur_d_ratio is: " + str(cur_d_ratio))
    for dp in range(0, dparts):
        lower_bound_ratio = float(dp) / float(dparts)
        upper_bound_ratio = float(dp+1) / float(dparts)
        if plog:
            print("lower_bound_ratio is: " + str(lower_bound_ratio))
            print("upper_bound_ratio is: " + str(upper_bound_ratio))
        if cur_d_ratio >= lower_bound_ratio and cur_d_ratio < upper_bound_ratio:
            cur_dp = dp
    vnorm_bins = vnorm_bins_all[cur_dp]
    if plog:
        print("cur_dp is: " + str(cur_dp))

    v_cur = [state[_xd], state[_yd]]
    v_next = [float(state[_xd] + action[0]), float(state[_yd] + action[1])]
    vnorm_cur = float(math.sqrt(state[_xd]**2 + state[_yd]**2))
    vnorm_next = float(math.sqrt(v_next[0]**2 + v_next[1]**2))

    # map velocity features:
    cur_vbin = 0
    for bin_num in range(0,len(vnorm_bins)):
        # if plog:
        #     print("vnorm_bins is: ")
        #     print(vnorm_bins)
        #     print("vnorm_bins is: ")
        #     print(vnorm_bins)
        if vnorm_cur >= vnorm_bins[bin_num,0] and vnorm_cur < vnorm_bins[bin_num,1]:
            feats[v0 + bin_num] = 1
            cur_vbin = bin_num
            if plog:
                print("state[_xd], state[_yd], v_cur is: " + str(state[_xd]) + ", " + str(state[_yd]) + ", " + str(v_cur))
                print("feats is: " + str(feats))
                # input("make sure this is right...")

    # # map acceleration/velocity interaction features:
    # vd_bins = np.array([[-np.inf,-1.5],[-1.5,-.5],[-.5, .5],[.5,1.5],[1.5,np.inf]])
    # # vd_bins = np.array([[-np.inf,-3.5],[-3.5,-1.5],[-1.5,-.5],[-.5, .5],[.5,1.5],[1.5,3.5],[3.5,np.inf]])
    # vnorm_delta = vnorm_next - vnorm_cur
    # for bin_num in range(0,len(vd_bins)):
    #     if vnorm_delta >= vd_bins[bin_num,0] and vnorm_delta < vd_bins[bin_num,1]:
    #         feats[int(v0_ss + (len(vd_bins) * cur_vbin) + bin_num)] = 1
    #         if plog:
    #             print("state is: " + str(state))
    #             print("action is: " + str(action))
    #             print("int(v0_ss + (len(vd_bins) * cur_vbin) + bin_num) is: " + str(v0_ss) + " + (" + str(len(vd_bins))  + " * " +  str(cur_vbin)  + ") + " + str(bin_num))
    #             print("feats is: ")
    #             print(feats)
    #             input("check acceleration-velocity interaction features")

    # angular motion style features (6):
    if plog:
        print("v_cur is: " + str(v_cur))
        print("v_next is: " + str(v_next))
    dot_prod = np.dot(v_cur,v_next)
    vnorm_cur = np.linalg.norm(v_cur)
    vnorm_next = np.linalg.norm(v_next)
    if (vnorm_cur * vnorm_next != 0):
        ratio = dot_prod / (vnorm_cur * vnorm_next)
        if ratio > 1.:
            ratio = 1.
        elif ratio < -1.:
            ratio = -1.
        angle = (180./np.pi) * math.acos(ratio)
    else:
        # print("vnorm_cur * vnorm_next == 0 !!!!!")
        angle = np.nan # DESQ do I want "nan" to make this feature be true, or no??? Need to think about impact..
    if plog:
        print("angle is: " + str(angle))
    if not np.isnan(angle):
        if angle >= 0. and angle < 30.:
            feats[v_ang_0] = 1
        elif angle >= 30. and angle < 60.:
            feats[v_ang_1] = 1
        elif angle >= 60. and angle < 90.:
            feats[v_ang_2] = 1
        elif angle >= 90. and angle < 120.:
            feats[v_ang_3] = 1
        elif angle >= 120. and angle < 150.:
            feats[v_ang_4] = 1
        elif angle >= 150. and angle < 180.:
            feats[v_ang_5] = 1

    # angular motion relative to goal features (6):
    v_goal = [float(goal[0]-state_next[_x]), float(goal[1]-state_next[_y])]
    v_action = [float(state_next_next[_x]-state_next[_x]), float(state_next_next[_y]-state_next[_y])]

    dot_prod = np.dot(v_goal,v_action)
    vnorm_goal = np.linalg.norm(v_goal)
    vnorm_action = np.linalg.norm(v_action)
    if (vnorm_goal * vnorm_action != 0):
        ratio = dot_prod / (vnorm_goal * vnorm_action)
        if ratio > 1.:
            ratio = 1.
        elif ratio < -1.:
            ratio = -1.
        angle = (180./np.pi) * math.acos(ratio)
    else:
        angle = np.nan

    if plog:
        print("angle for rel goal feature is: " + str(angle))
    if not np.isnan(angle):
        if angle >= 0. and angle < 30.:
            feats[g_ang_0] = 1
        elif angle >= 30. and angle < 60.:
            feats[g_ang_1] = 1
        elif angle >= 60. and angle < 90.:
            feats[g_ang_2] = 1
        elif angle >= 90. and angle < 120.:
            feats[g_ang_3] = 1
        elif angle >= 120. and angle < 150.:
            feats[g_ang_4] = 1
        elif angle >= 150. and angle < 180.:
            feats[g_ang_5] = 1

    #map  location features relative to goal (3):
    if state_next_dist_goal_euc - resulting_dist_goal_euc > 0:
        feats[closer_euc] = 50

    if (state_next_next[_x] >= goal[_x]-5) and (state_next_next[_x] <= goal[_x]+5) and (state_next_next[_y] >= goal[_y]-5) and (state_next_next[_y] <= goal[_y]+5):
        feats[in_goal] = 500

    # if cur_dist_goal_x - resulting_dist_goal_x > 0:
    #     feats[closer_x + (k*dp)] = 1
    # if cur_dist_goal_y - resulting_dist_goal_y > 0:
    #     feats[closer_y + (k*dp)] = 1


    # # location features relative to being out_of_bounds within the grid world (3):
    # if state_next_next[_x] >= x_min and state_next_next[_x] <= x_max and state_next_next[_y] >= y_min and state_next_next[_y] <= y_max:
    #     feats[ob0] = 1
    #     feats[ob1] = 1
    #     feats[ob2] = 1
    #     # feats[ob3] = 1
    #     # feats[ob4] = 1

    # location features relative to being out_of_bounds within the grid world (3):
    if state_next_next[_x] < x_min or state_next_next[_x] > x_max or state_next_next[_y] < y_min or state_next_next[_y] > y_max:
        feats[ob0] = 50
        # feats[ob1] = 1
        # feats[ob2] = 1
        # feats[ob3] = 1
        # feats[ob4] = 1


    if plog:
        print("cur_dp is:: " + str(cur_dp))
        print("at end of map_features()...feats is: " + str(np.shape(feats)))
        print(feats)
        input("pausing to review feats vector...")
    return feats, cur_dp

# RIGHT HERE! TODO next: ...then debug map_features, then add multiple controllers...

# def map_features(state, start, goal, xd_bins, yd_bins, action=None):
#     k = full_k
#     # plog = True
#     plog = False
#     flog = False
#     # feats = np.zeros(k, np.int)
#     feats = np.zeros(k*dparts, np.int)

#     cur_dist_goal_x = math.fabs(goal[0] - state[_x])
#     cur_dist_goal_y = math.fabs(goal[1] - state[_y])
#     cur_dist_goal_euc = math.sqrt((cur_dist_goal_x**2) + (cur_dist_goal_y**2))
#     state_next = sim.get_next_state(state, action)
#     # we don't care about the action here bc all we need is the x,y position from the projection of two states out in time:
#     state_next_next = sim.get_next_state(state_next, [0,0])
#     resulting_dist_goal_x = math.fabs(goal[0] - state_next_next[_x])
#     resulting_dist_goal_y = math.fabs(goal[1] - state_next_next[_y])
#     resulting_dist_goal_euc = math.sqrt((resulting_dist_goal_x**2) + (resulting_dist_goal_y**2))
#     if plog:
#         print("goal[x] - cur_state[_x] =  " + str(goal[0]) + " - "+ str(state[_x]))
#         print("goal[x] - state_next_next[_x] =  " + str(goal[0]) + " - "+ str(state_next_next[_x]))
#         print("dif_x = " + str(cur_dist_goal_x - resulting_dist_goal_x))
#         print("goal[y] - cur_state[_y] =  " + str(goal[1]) + " - "+ str(state[_y]))
#         print("goal[y] - state_next_next[_y] =  " + str(goal[1]) + " - "+ str(state_next_next[_y]))
#         print("dif_y = " + str(cur_dist_goal_y - resulting_dist_goal_y))
#         print("dif_euc = " + str(cur_dist_goal_euc - resulting_dist_goal_euc))

#     d_tot_euc = math.sqrt((start[_x] - goal[_x])**2 + (start[_y] - goal[_y])**2) # total euclidean distance from the start to the goal
#     cur_d_ratio = float(cur_dist_goal_euc) / float(d_tot_euc)
#     cur_d_ratio = 1.0 - cur_d_ratio
#     if cur_d_ratio < 0.:
#         cur_d_ratio = 0.
#     cur_dp  = -1
#     if plog:
#         print("cur_dist_goal_euc is: " + str(cur_dist_goal_euc))
#         print("d_tot_euc is: " + str(d_tot_euc))
#         print("cur_d_ratio is: " + str(cur_d_ratio))
#     for dp in range(0, dparts):
#         lower_bound_ratio = float(dp) / float(dparts)
#         upper_bound_ratio = float(dp+1) / float(dparts)
#         if plog:
#             print("lower_bound_ratio is: " + str(lower_bound_ratio))
#             print("upper_bound_ratio is: " + str(upper_bound_ratio))
#         if cur_d_ratio >= lower_bound_ratio and cur_d_ratio < upper_bound_ratio:
#             cur_dp = dp
#     if plog:
#         print("cur_dp is: " + str(cur_dp))
#     for dp in range(0, dparts):
#         # map velocity features:
#         if dp == cur_dp:
#             feats[dp_in + (k*dp)] = 1
#             for bin_num in range(0,len(xd_bins)):
#                 # if plog:
#                 #     print("xd_bins is: ")
#                 #     print(xd_bins)
#                 #     print("yd_bins is: ")
#                 #     print(yd_bins)
#                 #     print("state[_xd] is: " + str(state[_xd]))
#                 #     print("state[_yd] is: " + str(state[_yd]))
#                 #     input("check it out...")
#                 if math.fabs(state[_xd]) >= xd_bins[bin_num,0] and math.fabs(state[_xd]) < xd_bins[bin_num,1]:
#                     feats[(xd_0 + bin_num) + (k*dp)] = 1
#                     if plog:
#                         print("state[_xd] is: " + str(state[_xd]))
#                         print("xd_0 is: " + str(xd_0)  + "\t bin_num is: "  + str(bin_num))
#                         print("xd_0 + bin_num is: " + str(xd_0 + bin_num))
#                         print("feats is: " + str(feats))
#             for bin_num in range(0,len(yd_bins)):
#                 if math.fabs(state[_yd]) >= yd_bins[bin_num,0] and math.fabs(state[_yd]) < yd_bins[bin_num,1]:
#                     feats[(yd_0 + bin_num) + (k*dp)] = 1
#                     if plog:
#                         print("state[_yd] is: " + str(state[_yd]))
#                         print("yd_0 + bin_num is: " + str(yd_0 + bin_num))
#                         print("feats is: " + str(feats))
#             # xdd_bins = np.array([[-np.inf,-3.5],[-3.5,-1.5],[-1.5,-.5],[-.5, .5],[.5,1.5],[1.5,3.5],[3.5,np.inf]])
#             # ydd_bins = np.array([[-np.inf,-3.5],[-3.5,-1.5],[-1.5,-.5],[-.5, .5],[.5,1.5],[1.5,3.5],[3.5,np.inf]])
#             xdd_bins = np.array([[-np.inf,-1.5],[-1.5,-.5],[-.5, .5],[.5,1.5],[1.5,np.inf]])
#             ydd_bins = np.array([[-np.inf,-1.5],[-1.5,-.5],[-.5, .5],[.5,1.5],[1.5,np.inf]])
#             for bin_num in range(0,len(xdd_bins)):
#                 if action[0] >= xdd_bins[bin_num,0] and action[0] < xdd_bins[bin_num,1]:
#                     feats[(xdd_n2 + bin_num) + (k*dp)] = 1
#                     if plog:
#                         print("action[0] is: " + str(action[0]))
#                         print("xdd_n2 is: " + str(xdd_n2)  + "\t bin_num is: "  + str(bin_num))
#                         print("xdd_n2 + bin_num is: " + str(xdd_n2 + bin_num))
#             for bin_num in range(0,len(ydd_bins)):
#                 if action[1] >= ydd_bins[bin_num,0] and action[1] < ydd_bins[bin_num,1]:
#                     feats[(ydd_n2 + bin_num) + (k*dp)] = 1
#                     if plog:
#                         print("action[1] is: " + str(action[1]))
#                         print("ydd_n2 is: " + str(ydd_n2)  + "\t bin_num is: "  + str(bin_num))
#                         print("ydd_n2 + bin_num is: " + str(ydd_n2 + bin_num))
#             if plog:
#                 print("feats is: ")
#                 print(feats)

#             # location features relative to goal (3):
#             if cur_dist_goal_x - resulting_dist_goal_x > 0:
#                 feats[closer_x + (k*dp)] = 1
#             if cur_dist_goal_y - resulting_dist_goal_y > 0:
#                 feats[closer_y + (k*dp)] = 1
#             if cur_dist_goal_euc - resulting_dist_goal_euc > 0:
#                 feats[closer_euc + (k*dp)] = 1
#         # note this is outside the "if dp == cur_dp" check bc we ALWAYS want to punish being out of bound
#         # location features relative to being out_of_bounds within the grid world (1):
#         if state_next_next[_x] < x_min or state_next_next[_x] > x_max or state_next_next[_y] < y_min or state_next_next[_y] > y_max:
#             feats[out_of_bounds + (k*dp)] = 1
#         else:
#             feats[out_of_bounds + (k*dp)] = 0

#         if dp == cur_dp:
#             # motion style features (2):
#             if float(state[_xd])/float(state[_xd] + action[0] + .01) < 0:
#                 feats[abrupt_x + (k*dp)] = 1
#                 if plog:
#                     print("abrupt_x = TRUE")
#             if float(state[_yd])/float(state[_yd] + action[1] + .01) < 0:
#                 feats[abrupt_y + (k*dp)] = 1
#                 if plog:
#                     print("abrupt_y = TRUE")

#             # angular motion style features (2):
#             v_cur = [state[_xd], state[_yd]]
#             v_next = [float(state[_xd] + action[0]), float(state[_yd] + action[1])]
#             if plog:
#                 print("v_cur is: " + str(v_cur))
#                 print("v_next is: " + str(v_next))
#             dot_prod = np.dot(v_cur,v_next)
#             v_cur_norm = np.linalg.norm(v_cur)
#             v_next_norm = np.linalg.norm(v_next)
#             if (v_cur_norm * v_next_norm != 0):
#                 ratio = dot_prod / (v_cur_norm * v_next_norm)
#                 if ratio > 1.:
#                     ratio = 1.
#                 elif ratio < -1.:
#                     ratio = -1.
#                 angle = (180./np.pi) * math.acos(ratio)
#             else:
#                 # print("v_cur_norm * v_next_norm == 0 !!!!!")
#                 angle = 0 # DESQ do I want "nan" to make this feature be true, or no??? Need to think about impact..
#             if plog:
#                 print("angle is: " + str(angle))
#             if (angle >= 0. and angle < 30.) or (math.isnan(angle)): # DESQ do I want "nan" to make this feature be true, or no??? Need to think about impact..
#                 feats[v_ang_0 + (k*dp)] = 1
#             elif angle >= 30. and angle < 60.:
#                 feats[v_ang_1 + (k*dp)] = 1
#             elif angle >= 60. and angle < 90.:
#                 feats[v_ang_2 + (k*dp)] = 1
#             elif angle >= 90. and angle < 120.:
#                 feats[v_ang_3 + (k*dp)] = 1
#             elif angle >= 120. and angle < 150.:
#                 feats[v_ang_4 + (k*dp)] = 1
#             elif angle >= 150. and angle < 180.:
#                 feats[v_ang_5 + (k*dp)] = 1

#     if plog:
#         print("at end of map_features()...feats is: " + str(np.shape(feats)))
#         print(feats)
#         # input("pausing")
#     # feats = feats[closer_euc:in_bounds+1]
#     # feats = feats[closer_x:out_of_bounds+1]
#     # feats = feats[in_bounds]
#     if plog:
#         print("at end of map_features()...reduced feats is: " + str(np.shape(feats)))
#         print(feats)
#         # input("holding...")
#     return feats

def compute_Efeats_exp(demos, starts, goals, vnorm_bins): # demos is a list of 2d ndarrays
    plog = False
    flog = False
    # computer feature_expectations of "expert":
    M = len(demos)
    tot_T = 0.
    avg_V = 0.
    m = 0
    for demo in demos:
        # print(demo)
        demo_dist = math.sqrt(((goals[m][0]-starts[m][0])**2) + ((goals[m][1]-starts[m][1])**2))
        demo_time = float(len(demo[:,0]))
        demo_avg_V = demo_dist / demo_time
        avg_V += demo_avg_V
        tot_T = tot_T + demo_time
        m += 1
    avg_T = tot_T / float(M)
    avg_V = avg_V / float(M)

    # total_feats = np.zeros(k)
    total_feats = np.zeros((dparts, k))
    total_feats_debug = np.zeros((dparts,M,k)) #DEBUG
    demo_idx = 0
    for demo in demos:
        for t in range(0,len(demo)):
            demo_state = demo[t,:]
            state = demo_state[0:state_dim]
            if plog:
                print("state is: " + str(state))
            action = demo_state[4:6]
            if plog:
                print("action is: " + str(action))
                # input("in compute_Efeats_exp")
            features, dp = map_features(state, starts[demo_idx], goals[demo_idx], vnorm_bins, action)
            # print("features for expert is: ")
            # print(features)
            # input("what's wrong...")
            total_feats[dp] += (math.pow(discount, t) * features) # note that these are both vectors (i.e. 1d ndarrays), not scalars
            total_feats_debug[dp,demo_idx] += (math.pow(discount, t) * features) # note that these are both vectors (i.e. 1d ndarrays), not scalars
        demo_idx += 1
    for dp in range(0,dparts):
        total_feats[dp] = total_feats[dp] / M
        total_feats[dp] = total_feats[dp] / np.linalg.norm(total_feats[dp])
    Efeats_expert = total_feats
    if plog:
        print("Efeats_expert is: " + str(np.shape(Efeats_expert)))
        print(Efeats_expert)
        print("total_feats_debug is: " + str(np.shape(total_feats_debug)))
        print(total_feats_debug)
    return Efeats_expert, avg_T, avg_V

def compute_Efeats(iteration, pweights, start_dontuse, goal_dontuse, vnorm_bins, T):
    # plog = True
    plog = False
    flog = False
    Efeats = np.zeros((dparts,k))
    for m in range(0,num_rollouts):
        Efeats_m = np.zeros((dparts,k))
        s = np.zeros(state_dim)
        # s[_x] = start[0]
        # s[_y] = start[1]
        start = np.zeros(2)
        goal = np.zeros(2)
        start[_x] = np.random.random_integers(x_min, x_max)
        start[_y] = np.random.random_integers(y_min, y_max)
        goal[_x] = np.random.random_integers(x_min, x_max)
        goal[_y] = np.random.random_integers(y_min, y_max)
        # s[_x] = np.random.random_integers(x_min, x_max)
        # s[_y] = np.random.random_integers(y_min, y_max)
        s[_x] = start[_x]
        s[_y] = start[_y]
        s[_xd] = 0.
        s[_yd] = 0.

        termination = False
        t = 0
        # dp = 0
        while not termination and t < max_iters_per_episode:
            if pweights == None:
                a = global_actions[np.random.random_integers(0, (num_actions**2) - 1)]
                features, dp = map_features(s, start, goal, vnorm_bins, a)
            else:
                a, idx_a, Qsa, Fsa, dp =  select_action_from_policy(pweights, s, 0., start, goal, vnorm_bins)
                features = Fsa[0:-1]
            if plog:
                print("selected action is: " + str(a))
            Efeats_m[dp] += (math.pow(discount, t) * features) # note that these are both vectors (i.e. 1d ndarrays), not scalars
            s = sim.get_next_state(s,a)
            if plog:
                print("\tt: " + str(t) + "\tnext state is: " + str(s))
            termination, termination_result, termination_reason, termination_details = is_state_terminal(s, goal)
            t += 1
        # Efeats_m_norm = np.linalg.norm(Efeats_m)
        # Efeats_m = Efeats_m / Efeats_m_norm
        Efeats[dp] += Efeats_m[dp]
    for dp in range(0,dparts):
        Efeats[dp] = Efeats[dp] / num_rollouts
        temp_norm = np.linalg.norm(Efeats[dp])
        if temp_norm > 0:
            Efeats[dp] = Efeats[dp] / temp_norm
    # print("iteration is: " + str(iteration))
    # print("Efeats is: ")
    # print(Efeats)
    # input("checj to make sure compute_Efeats() is working correctly with multicontroller implemenation")
    return Efeats, t

def determine_best_policy(policies, Efeats, Efeats_e):
    # print("len(policies) is: " +str(len(policies)))
    # print("np.shape(policies) is: " +str(np.shape(policies)))
    # print("len(Efeats) is: " +str(len(Efeats)))
    # print("np.shape(Efeats) is: " +str(np.shape(Efeats)))
    # print("len(Efeats_e) is: " +str(len(Efeats_e)))
    # print("np.shape(Efeats_e) is: " +str(np.shape(Efeats_e)))
    deltas = np.zeros((dparts,len(Efeats)))
    for dp in range(0,dparts):
        for i in range(0, len(Efeats)):
            deltas[dp,i] = np.linalg.norm(Efeats_e[dp] - Efeats[i][dp])
    best_policy_idxs_all = np.zeros(dparts, np.int)
    for dp in range(0,dparts):
        best_policy_idxs = np.argwhere(deltas[dp] == np.amin(deltas[dp]))
        # print("best_policy_idxs for dp=" +str(dp) + " is: " + str(np.shape(best_policy_idxs)))
        # print(best_policy_idxs)
        best_policy_idxs = best_policy_idxs.flatten()

        if len(best_policy_idxs) == 1:
            pass
            # input("NOT multiple BEST POLICIES!!! review...")
        else:
            pass
            # input("MULTIPLE BEST POLICIES!!! review...")

        if best_policy_idxs[0] == 0:
            # print("best policy is RANDOM, so setting to policy 0")
            best_policy_idxs_all[dp] = 0
        else:
            best_policy_idxs_all[dp] = best_policy_idxs[0] - 1

    # print("best_policy_idxs_all is: " + str(np.shape(best_policy_idxs_all)))
    # print(best_policy_idxs_all)
    # input("review new function...")

    best_pweights = np.zeros((dparts,k+1))
    best_rweights = np.zeros((dparts,k))
    for dp in range(0, dparts):
        # print("policies is: " + str(np.shape(policies)))
        # print(policies)
        # input(".............")
        # print("best_policy_idxs_all is: ")
        # print(best_policy_idxs_all)
        # input(".............")
        # print("best_policy_idxs_all[dp] is: ")
        # print(best_policy_idxs_all[dp])
        # input(".............")


        p = policies[best_policy_idxs_all[dp]]
        # print("p is: ")
        # print(p)
        # input(".............")
        p_pweights = p[0]
        p_rweights = p[1]
        # print("p_pweights is: ")
        # print(p_pweights)
        # input(".............")
        p_pweights_dp = p_pweights[dp]
        p_rweights_dp = p_rweights[dp]
        # print("p_pweights_dp is: ")
        # print(p_pweights_dp)
        # input(".............")

        # best_pweights[dp] = policies[best_policy_idxs_all[dp]][0][dp]
        best_pweights[dp] = p_pweights_dp
        # best_rweights[dp] = policies[best_policy_idxs_all[dp]][1][dp]
        best_rweights[dp] = p_rweights_dp
    best_policy = (best_pweights, best_rweights)
    return best_policy, best_policy_idxs_all

def IRL_initialize(demos, starts, goals, vnorm_bins):
    # compute feature_expectations of the expert
    Efeats_exp, avg_T, avg_V = compute_Efeats_exp(demos, starts, goals, vnorm_bins)
    return  Efeats_exp, avg_T, avg_V



##################################################################################
##################################################################################
##################################################################################
##################################################################################
def run_IRL(exact):
    with open(logfile_name, "w") as logfile:

        demos, starts, goals, modes, emos = process_demos(filenames)
        start = starts[0]
        goal = goals[0]
        mode = modes[0]
        emo = emos[0]


        vnorm_bins = detect_bins(demos, starts, goals) # TODO: make sure this is working correctly (now updated for multicontroller)
        print("vnorm_bins is: " + str(np.shape(vnorm_bins)))
        print(vnorm_bins)
        # input("check and make sure that vnorm_bins looks right...")


        # 0. Initialize by computing feature expectations for Expert Policy:
        Efeats_e, T, avg_V = IRL_initialize(demos, starts, goals, vnorm_bins)
        T = int(T)
        logfile.write("T is: " + str(T))
        logfile.write("\n")
        logfile.write("Efeats_e is:" + str(np.shape(Efeats_e)))
        logfile.write("\n")
        logfile.write(str(Efeats_e))
        logfile.write("\n")
        # print("T is: " + str(T))
        # print("Efeats_e is:" + str(np.shape(Efeats_e)))
        # print(Efeats_e)
        # input("compute_Efeats_exp() E_exp")

        # 1. Randomly pick some policy π^(0), compute (or approximate via Monte Carlo) µ^(0) = µ(π^(0)), and set i = 1.

        # pweights_initial = np.random.uniform(-.1,.1,k*dparts+1)
        # pweights_initial_norm = np.linalg.norm(pweights_initial)
        # pweights_initial = pweights_initial / pweights_initial_norm
        Efeats_0, t_ignore = compute_Efeats(0, None, start, goal, vnorm_bins, T)

        logfile.write("Efeats_0 is:" + str(np.shape(Efeats_0)))
        logfile.write("\n")
        logfile.write(str(Efeats_0))
        logfile.write("\n")
        # print("Efeats_0 is:" + str(np.shape(Efeats_0)))
        # print(Efeats_0)
        # input("compute_Efeats() E_0")

        paths = list()
        Efeats = list()
        Efeats_bar = list()
        policies = list()
        margins = np.zeros(dparts)
        margins.fill(np.inf)
        rweights = np.zeros((dparts, k))

        # Efeats.append(Efeats_e)
        Efeats.append(Efeats_0)
        Efeats_bar.append(Efeats_0)
        i = 1
        logfile.write("******************************************************************************************************\n\n")
        logfile.write("\n")
        # print("******************************************************************************************************\n\n")

        while(np.linalg.norm(margins) > epsilon_IRL and i < max_iters):
            # 2. Compute t^(i) = maxw:kwk2≤1 minj∈{0..(i−1)} wT(µE −µ(j)), and let w(i) be the value of w that attains this maximum.
            # QP or SVM solver:
            # margin, rweights = optimize_reward_weights(Efeats) # TODO: not sure if/how well this function is working...appears to be working fine but will have to run multiple iterations of the outer (IRL) loop to really be able to tell, thus need to handle RL solver first...

#############################################################3
#############################################################3
            if use_svm:
                # margin, rweights = optimize_reward_weights_svm(Efeats, Efeats_e)
                for dp in range(0,dparts):
                    Efeats_array = np.array(Efeats)
                    margins[dp], rweights[dp] = optimize_reward_weights_svm(Efeats_array[:,dp], Efeats_e[dp])
            else:
                margin, rweights, Efeats_bar_iprev = optimize_reward_weights_projection(Efeats, Efeats_bar, Efeats_e)  #TODO: DEBUG this function...
                Efeats_bar.append(Efeats_bar_iprev)
#############################################################3
#############################################################3

            logfile.write("margins is: " + str(margins))
            logfile.write("\n")
            logfile.write("rweights is: ")
            logfile.write("\n")
            logfile.write(str(rweights))
            logfile.write("\n")
            print("margins is: " + str(margins))
            print("rweights is: " + str(rweights))
            if i % 1 ==0:
                pass
                # input("pausing after optimize_reward_weights()")
            # break
            # 4. Using the RL algorithm, compute the optimal policy π^(i) for the MDP using rewards R = (w^(i))^T φ.
            # RL solver:
            # pweights = SARSA(rweights, start, goal, vnorm_bins, i)
            pweights = np.zeros((dparts,k+1))
            for dp in range(0,dparts):
                pweights[dp,0:k] = rweights[dp,:]
                pweights[dp,-1] = 0.
            policies.append((pweights, rweights))
            if i % 1 ==0:
                pass
                # input("pausing after policy = SARSA():")
            ########################################################33
            #SAVE STUFF TO FILE FOR ANALYSIS
            ########################################################33
            path, err_goal, termination_msg, path_duration = create_path_from_policy(pweights, start, goal, vnorm_bins, T)
            paths.append(path)
            save_pweights = str(demo_group) + "pweights_" + str(i)
            save_rweights = str(demo_group) + "rweights_" + str(i)
            save_pweights_cur_opt = "_opt" + str(demo_group) + "_pweights_" + str(i)
            save_rweights_cur_opt = "_opt" + str(demo_group) + "_rweights_" + str(i)
            save_weights_to_file_numpy(save_pweights, pweights)

            logfile.write("\n\n----------------------------------------------------------------------------------------------------------------------")
            logfile.write("\n")
            logfile.write("path is: ")
            logfile.write("\n")
            logfile.write(str(path))
            logfile.write("\n")
            logfile.write("ct: \t" + str(i) + "\tpath_duration is: " + str(path_duration) + "\terr_goal is: " + str(err_goal))
            logfile.write("\n")
            logfile.write("rweights is: ")
            logfile.write("\n")
            logfile.write(str(rweights))
            logfile.write("\n")
            logfile.write("pweights is: ")
            logfile.write("\n")
            logfile.write(str(pweights))
            logfile.write("\n")
            logfile.write(str(termination_msg))
            logfile.write("\n")

            save_paths_to_file([path], start, goal, emo, mode, save_recreates, i)
            #######################################################################
            #######################################################################

            # 5. Compute (or estimate) µ^(i) = µ(π^(i)).
            # compute feature expecations of new policy:
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            Efeats_i, T_term = compute_Efeats(i, pweights, start, goal, vnorm_bins, T)

            Efeats.append(Efeats_i)
            logfile.write("i is: " + str(i) + "\tT_term is: " + str(T_term) + "\tEfeats_i is: ")
            logfile.write("\n")
            logfile.write(str(Efeats_i))
            logfile.write("\n")
            print("i is: " + str(i) + "\tT_term is: " + str(T_term) + "\tEfeats_i is: ")
            print(Efeats_i)
            if i % 1 ==0:
                pass
                # input("pausing after Efeats_i, T_term = compute_Efeats()")

            cur_best_policy, cur_best_policy_idxs = determine_best_policy(policies, Efeats, Efeats_e)
            save_weights_to_file_numpy(save_pweights_cur_opt, cur_best_policy[0])

            cur_best_path, err_goal, termination_msg, path_duration = create_path_from_policy(cur_best_policy[0], start, goal, vnorm_bins, T)
            save_paths_to_file([cur_best_path], start, goal, emo, mode, save_recreates_cur_opt, i)

            logfile.write("\n\n")
            logfile.write("cur_best_path is: ")
            logfile.write("\n")
            logfile.write(str(cur_best_path))
            logfile.write("\n")
            logfile.write("cur_best_policy_idxs is: ")
            logfile.write("\n")
            logfile.write(str(cur_best_policy_idxs))
            logfile.write("\n")
            logfile.write("ct: \t" + str(i) + "\tcur_best_path_duration is: " + str(path_duration) + "\tcur_best_err_goal is: " + str(err_goal))
            logfile.write("\n")
            logfile.write("cur_best_rweights is: ")
            logfile.write("\n")
            logfile.write(str(cur_best_policy[1]))
            logfile.write("\n")
            logfile.write("cur_best_pweights is: ")
            logfile.write("\n")
            logfile.write(str(cur_best_policy[0]))
            logfile.write("\n")
            logfile.write(str(termination_msg))
            logfile.write("\n")

            # 6. Set i = i + 1, and go back to step 2.
            i += 1
            logfile.write("IRL iteration # " + str(i) + "completed")
            logfile.write("\n")
            logfile.write("************************************************************************************\n\n")
            logfile.write("\n")
            print("IRL iteration # " + str(i) + "completed")
            print("************************************************************************************\n\n")

        logfile.write("final margin is: " + str(margins))
        logfile.write("\n")
        print("final margin is: " + str(margins))
        # input("pausing:")
        return policies, Efeats, Efeats_e
    # ###################################################

exact = False
# policies, Efeats, Efeats_e = run_IRL(exact)

# opt_policy, opt_policy_idx = determine_best_policy(policies, Efeats, Efeats_e)
# print("at very end of all this...opt_policy_idx is: " + str(opt_policy_idx))
# print(opt_policy)
# input("OVER as soon as you press key...")








