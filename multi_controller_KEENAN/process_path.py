import scipy as sci
import numpy as np
import os
import config as c
import time
import math
import matplotlib.pyplot as plt

def compute_derivatives(feats, T, start, goal):
    x_ = 0
    y_ = 1
    # dist_from_start_euc_ = 2
    # dist_from_start_x_ = 3
    # dist_from_start_y_ = 4
    # dist_from_goal_euc_ = 5
    # dist_from_goal_x_ = 6
    # dist_from_goal_y_ = 7
    x_dot_ = 2
    y_dot_ = 3
    x_dot_dot_ = 4
    y_dot_dot_ = 5

    # x_ = 0
    # y_ = 1
    # dist_from_start_euc_ = 2
    # dist_from_start_x_ = 3
    # dist_from_start_y_ = 4
    # dist_from_goal_euc_ = 5
    # dist_from_goal_x_ = 6
    # dist_from_goal_y_ = 7
    # x_dot_ = 8
    # x_dot_dot_ = 10
    # y_dot_ = 9
    # y_dot_dot_ = 11

    print("start is: " + str(start))
    print("goal is: " + str(goal))
    print("T is: " + str(T))

    for t in range(0, T): # T = len(feats[0,:])
        # dist_from_start_euc = math.sqrt(math.pow(start[0]-feats[x_,t], 2) + math.pow(start[1]-feats[y_,t], 2))
        # feats[dist_from_start_euc_,t] = dist_from_start_euc

        # dist_from_start_x = math.fabs(start[0]-feats[x_,t])
        # feats[dist_from_start_x_,t] = dist_from_start_x

        # dist_from_start_y = math.fabs(start[1]-feats[y_,t])
        # feats[dist_from_start_y_,t] = dist_from_start_y

        # dist_from_goal_euc = math.sqrt(math.pow(goal[0]-feats[x_,t], 2) + math.pow(goal[1]-feats[y_,t], 2))
        # feats[dist_from_goal_euc_,t] = dist_from_goal_euc

        # dist_from_goal_x = math.fabs(goal[0]-feats[x_,t])
        # feats[dist_from_goal_x_,t] = dist_from_goal_x

        # dist_from_goal_y = math.fabs(goal[1]-feats[y_,t])
        # feats[dist_from_goal_y_,t] = dist_from_goal_y

        if t == 0:
            # x_dot and y_dot for t == 0
            x_dot = (feats[t+1,x_]-feats[t,x_]) # v_t = (x_t+1 - x_t)
            feats[t,x_dot_] = x_dot

            y_dot = (feats[t+1,y_]-feats[t,y_]) # v_t = (y_t+1 - y_t)
            feats[t,y_dot_] = y_dot

        elif t > 0 and t < T-1:
            x_dot = (feats[t+1,x_]-feats[t,x_]) # v_t = (x_t+1 - x_t)
            feats[t,x_dot_] = x_dot

            x_dot_dot_prev = feats[t,x_dot_]-feats[t-1,x_dot_] # a_t-1 = v_t - v_t-1
            feats[t-1,x_dot_dot_] = x_dot_dot_prev

            y_dot = (feats[t+1,y_]-feats[t,y_]) # v_t = (y_t+1 - y_t)
            feats[t,y_dot_] = y_dot

            y_dot_dot_prev = feats[t,y_dot_]-feats[t-1,y_dot_] # a_t-1 = v_t - v_t-1
            feats[t-1,y_dot_dot_] = y_dot_dot_prev

        elif t == T-1:
            x_dot = 0 # assumes that the final velocity is 0
            feats[t,x_dot_] = x_dot

            y_dot = 0 # assumes that the final velocity is 0
            feats[t,y_dot_] = y_dot

            x_dot_dot_prev = feats[t,x_dot_]-feats[t-1,x_dot_] # a_t-1 = v_t - v_t-1
            feats[t-1,x_dot_dot_] = x_dot_dot_prev

            y_dot_dot_prev = feats[t,y_dot_]-feats[t-1,y_dot_] # a_t-1 = v_t - v_t-1
            feats[t-1,y_dot_dot_] = y_dot_dot_prev

            feats[t,x_dot_dot_] = 0 # assumes that the final acceleration is 0
            feats[t,y_dot_dot_] = 0 # assumes that the final acceleration is 0

    return feats

def process_path(path, start, goal):
    T = len(path)
    # processed_path for reward learning:
        # euclidean distance from start
        # x distance from start
        # y distance from start
        # euclidean distance from goal
        # x distance from goal
        # y distance from goal
        # x_dot
        # y_dot
        # x_dot_dot
        # y_dot_dot
    # processed_path = np.zeros((12,T))
    processed_path = np.zeros((T,6))
    for t in range(0, T): # T = len(path)
        # get the x-coord
        processed_path[t,0] = path[t][0]
        # get the y-coord
        processed_path[t,1] = path[t][1]
    processed_path = compute_derivatives(processed_path, T, start, goal)
    # processed_path = processed_path[2:,:]
    return processed_path

def process_raw_path_from_file(filename):
    with open(filename, "r") as f:
        mode = (f.readline().split(":")[1]).rstrip()
        emo = (f.readline().split(":")[1]).rstrip()

        temp = (f.readline().split(":")[1]).rstrip().split(",")
        start_list = list()
        for string in temp:
            start_list.append(float(string))
        start_x = start_list[0] + (float(start_list[2] - start_list[0])/2.0)
        start_y = start_list[1] + (float(start_list[3] - start_list[1])/2.0)
        start = (start_x, start_y)

        temp = f.readline().split(":")[1].split(",")
        goal_list = list()
        for string in temp:
            goal_list.append(float(string))
        goal_x = goal_list[0] + (float(goal_list[2] - goal_list[0])/2.0)
        goal_y = goal_list[1] + (float(goal_list[3] - goal_list[1])/2.0)
        goal = (goal_x, goal_y)

        path = list()
        for line in f:
            x = float(line.split(",")[0].rstrip())
            y = float(line.split(",")[1].rstrip())
            path.append((x,y))

        return mode, emo, start, goal, path

# fname = "demo0.csv"
# fname_feats = "demo0_processed.csv"
# fname = "demo1.csv"
# fname_feats = "demo1_processed.csv"
# fname = "demo2.csv"
# fname_feats = "demo2_processed.csv"
# fname = "happy0.csv"
# fname_feats = "happy0_processed.csv"

def execute(target_filename_, save_to_file):
    suffix = ".csv"
    target_filename = target_filename_ + suffix
    output_filename = target_filename_ + "_processed" + suffix

    mode, emo, start, goal, path = process_raw_path_from_file(target_filename)
    processed_path = process_path(path, start, goal)

    # print("processed_path is: ")
    # print(processed_path)
    t_ = np.ones(len(processed_path[0:100,0]))
    for t in range(0,len(t_)):
        t_[t] = t
    plt.plot( t_, processed_path[0:100,0]/50.0, "b",  t_, processed_path[0:100,2], "g",  t_, processed_path[0:100,4], "r")
    # plt.show()
    plt.plot( t_, processed_path[0:100,1]/50.0, "b",  t_, processed_path[0:100,3], "g",  t_, processed_path[0:100,5], "r")
    # plt.show()

    if save_to_file:
        with open(output_filename, "w") as f:
            for t in range(0, len(processed_path[:,0])):
                for feat in range(0, len(processed_path[0,:])):
                    f.write(str(processed_path[t,feat]))
                    f.write(",")
                f.write("\n")

    return processed_path, start, goal, mode, emo



    ###########################


        # else:
        #     x_dot = 2.0*(feats[x_,t]-feats[x_,t-1]) - feats[x_dot_,t-1] # v_t = 2(x_t - x_t-1) - v_t-1
        #     x_dot_dot_prev = feats[x_dot_,t]-feats[x_dot_,t-1] # a_t-1 = v_t - v_t-1
        #     y_dot = 2.0*(feats[y_,t]-feats[y_,t-1])-feats[y_dot_,t-1] # v_t = 2(x_t - x_t-1) - v_t-1
        #     y_dot_dot_prev = feats[y_dot_,t]-feats[y_dot_,t-1] # a_t-1 = v_t - v_t-1
        #     feats[x_dot_,t] = x_dot
        #     feats[y_dot_,t] = y_dot
        #     feats[x_dot_dot_,t-1] = x_dot_dot_prev
        #     feats[y_dot_dot_,t-1] = y_dot_dot_prev
        # if t == T-1:
        #     x_dot_dot = 0-feats[x_dot_,t-1] # a_final = 0 - v_T-1
        #     y_dot_dot = 0-feats[y_dot_,t-1] # a_final = 0 - v_T-1
        #     feats[x_dot_dot_,t] = x_dot_dot
        #     feats[y_dot_dot_,t] = y_dot_dot












#############################

    # def compute_derivatives(waypoints_x, T, dt, x_dot_start, x_dot_dot_start):
    # # num_coords = len(waypoints[0,:])
    # waypoints = np.zeros((3,T))
    # waypoints[0,:] = waypoints_x
    # waypoints_x_dot = np.zeros(T)
    # waypoints_x_dot_dot = np.zeros(T)

    # for t in xrange(0,T):
    #     if(t==0):
    #         waypoints_x_dot[0] = x_dot_start
    #     elif(t==T-1):
    #         waypoints_x_dot[T-1] = 0
    #     else:
    #         waypoints_x_dot[t] = (waypoints_x[t] - waypoints_x[t-1])/dt
    #         # waypoints_x_dot[t] = (-1*waypoints_x_dot[t]) + 2*(waypoints_x[t] - waypoints_x[t-1])/dt

    # for t in xrange(0,T):
    #     if(t==0):
    #         waypoints_x_dot_dot[0] = x_dot_dot_start
    #     elif(t==T-1):
    #         waypoints_x_dot_dot[T-1] = 0
    #     else:
    #         waypoints_x_dot_dot[t] = (waypoints_x_dot[t] - waypoints_x_dot[t-1])/dt
    # waypoints[1,:] = waypoints_x_dot
    # waypoints[2,:] = waypoints_x_dot_dot
    # # plt.plot(waypoints[0], "b", waypoints[1], "g", waypoints[2], "r")
    # # plt.show()
    # return waypoints











