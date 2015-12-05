# scratch stuff that I deleted from rl.py (in proj2_c), but want to keep around for later use in case I need it...
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################

# demo_group = "___A_stoch_demo"
# filenames = ["JAGGED_demo0",
#                     "JAGGED_demo1",
#                     "JAGGED_demo2",
#                     "JAGGED_demo3",
#                     "JAGGED_demo4",
# ]
# filenames = ["A_demo0",
#                     "A_demo1",
#                     "A_demo2",
#                     "A_demo3",
#                     "A_demo4",
# ]
# filenames = ["demo0_processed.csv",
#                     "demo1_processed.csv",
#                     "demo2_processed.csv",
# ]
# save_name = "___A_stoch_recreates"
# emo = "Happy"
# mode = "Playback"

# start = [30.,390.]
# goal = [487.,229.]


def normalize_demo_lengths(demos):
    normalized_demos = list()
    tot_len = 0
    for demo in demos:
        tot_len += len(demo[:,0])
        # print(len(demo[:,0]))
    avg_len = math.floor(float(tot_len)/len(demos))
    # print("avg_len is: " + str(avg_len))
    # input("pausing outer")

    k_val = 5
    for demo in demos:
        normalized_demo = np.zeros((avg_len,demo_state_dim))
        T = len(demo[:,0])
        # print("T is: " + str(T))

        t = np.arange(T)
        for i in range(0,len(demo[0,:])):
            t_norming = np.arange(0,T,float(T)/avg_len)
            # print("len(t_norming) is: " + str(len(t_norming)))
            tck = interpolate.splrep(t, demo[:,i], k=k_val)
            normalized = interpolate.splev(t_norming, tck, der=0) # THIS IS THE KEY!!!
            # print("normalized is: ")
            # print(np.shape(normalized))
            # print(normalized)
            normalized_demo[:,i] = normalized
            # input("pausing inner")
        normalized_demos.append(normalized_demo)
    return normalized_demos

def Qfsa_map_idx(Fsa, xd_bins, yd_bins):
    #TODO: DEBUG this and make sure it works...
    idx_dp = 0
    idx_xdd = 0
    idx_ydd = 0
    idx_action = 0
    idx_xd = 0
    idx_yd = 0
    idx_closer_x = 0
    idx_closer_y = 0
    idx_closer_euc = 0
    idx_out_of_bounds = 0
    idx_ang = 0

    for dp in range(0, dparts):
        if Fsa[dp_in + (k*dp)] == 1:
            idx_dp = dp

    for bin_num in range(0,len(xd_bins)):
        if Fsa[(xd_0 + bin_num) + (k*dp)] == 1:
            idx_xd = bin_num
    for bin_num in range(0,len(yd_bins)):
        if Fsa[(yd_0 + bin_num) + (k*dp)] == 1:
            idx_yd = bin_num
    for bin_num in range(0,num_actions):
        if Fsa[(xdd_n2 + bin_num) + (k*dp)] == 1:
            idx_xdd = bin_num
    for bin_num in range(0,num_actions):
        if Fsa[(ydd_n2 + bin_num) + (k*dp)] == 1:
            idx_ydd = bin_num
    idx_action = idx_xdd*num_actions + idx_ydd
    idx_closer_x = Fsa[closer_x + (k*dp)]
    idx_closer_y = Fsa[closer_y + (k*dp)]
    idx_closer_euc = Fsa[closer_euc + (k*dp)]
    idx_out_of_bounds = Fsa[out_of_bounds + (k*dp)]

    for bin_num in range(0, k_angular_motion_style):
        if Fsa[v_ang_0 +bin_num + (k*dp)] == 1:
            idx_ang = bin_num

    idxs = [idx_action, idx_dp, idx_xd, idx_yd, idx_closer_x, idx_closer_y, idx_closer_euc, idx_out_of_bounds, idx_ang]
    return idxs

def Qfsa_map_feats(idxs, xd_bins, yd_bins):
    #TODO: DEBUG:
    idx_action = idxs[0]
    cur_dp = idxs[1]
    cur_ydd_bin = idx_action % num_actions
    cur_xdd_bin = int((idx_action - cur_ydd_bin)/ num_actions)
    cur_xd_bin = idxs[2]
    cur_yd_bin = idxs[3]
    cur_closer_x =  idxs[4]
    cur_closer_y =  idxs[5]
    cur_closer_euc =  idxs[6]
    cur_out_of_bounds =  idxs[7]
    cur_ang_bin =  idxs[8]

    feats = np.zeros(k*dparts, np.int)

    feats[dp_in + (k*cur_dp)] = 1
    feats[(xd_0 + cur_xd_bin) + (k*cur_dp)] = 1
    feats[(yd_0 + cur_yd_bin) + (k*cur_dp)] = 1
    feats[(xdd_n2 + cur_xdd_bin) + (k*cur_dp)] = 1
    feats[(ydd_n2 + cur_ydd_bin) + (k*cur_dp)] = 1
    feats[closer_x + (k*cur_dp)] = cur_closer_x
    feats[closer_y + (k*cur_dp)] = cur_closer_y
    feats[closer_euc + (k*cur_dp)] = cur_closer_euc
    feats[v_ang_0 +cur_ang_bin + (k*cur_dp)] = 1
    if cur_out_of_bounds:
        for dp in range(0,dparts):
            feats[out_of_bounds + (k*dp)] = cur_out_of_bounds
    # print("feats from Qfsa_map_feats() is: " + str(np.shape(feats)))
    # print(feats)
    return feats

def SARSA_exact(rweights, start, goal, xd_bins, yd_bins, IRL_iter, on_policy):
    # NOTES:
        # s = "state"
        # s_ = "next state"
        # a = "action"
        # a_ = "next action"
        # Fsa = "features mapping of state, s, and action, a"
        # Fsa_ = "features mapping of state, s_, and action, a_"
    # Qfsa is our feature-mapped state-action Q value matrix...every state can be boiled down to one of the entries in this array:
    Qfsa = np.zeros((num_actions**2, dparts, len(xd_bins), len(yd_bins), 2, 2, 2, 2, k_angular_motion_style))
    inner_ct = 0
    for a in range(0, len(global_actions**2)):
        for dp in range(0, dparts):
            for xd in range(0, len(xd_bins)):
                for yd in range(0, len(yd_bins)):
                    for closer_x in range(0, 2):
                        for closer_y in range(0, 2):
                            for closer_euc in range(0, 2):
                                for out_of_bounds in range(0, 2):
                                    for ang in range(0, k_angular_motion_style):
                                        idxs = [a,dp,xd,yd,closer_x,closer_y,closer_euc,out_of_bounds,ang]
                                        Fsa = Qfsa_map_feats(idxs, xd_bins, yd_bins)
                                        r = np.dot(rweights, Fsa)
                                        Qfsa[tuple(idxs)] = r
                                        inner_ct += 1
                                        # print("initializing policy... inner_ct is: " + str(inner_ct) + "\tr[" + str(idxs) + "]=" + str(r))
                                        # print("initializing policy... inner_ct is: " + str(inner_ct) + "\tr=" + str(r))

    # Qfsa_map_idx(Fsa, xd_bins, yd_bins): # note that if the DP approach works by iterating through the entire array,
    # as opposed to simulating rollouts, then this won't be needed until the actual "create_path_from_policy function...if at all???"
    # input("Finished initializing policy...")

    dif = np.inf
    ct = 0
    episode = 0
    _Qfsa = np.copy(Qfsa)
    while dif > epsilon_RL_convergence: # multiple dynamic programming loops:
        s = np.zeros(6)
        s[_x] = start[0] + np.random.random_integers(-10,10)
        s[_y] = start[1] + np.random.random_integers(-10,10)
        # s[_x] = np.random.random_integers(x_min, x_max)
        # s[_y] = np.random.random_integers(y_min, y_max)
        s[_xd] = 0.
        s[_yd] = 0.
        a, idx_a, Qsa, Fsa =  select_action_from_policy_exact(Qfsa, s, 0., start, goal, xd_bins, yd_bins)
        # a, idx_a, Qsa, Fsa =  select_action_from_policy_exact(Qfsa, s, epsilon, start, goal, xd_bins, yd_bins)

        termination = False # iterate through each step of a given episode
        t = 0
        while not termination:
            lr = lr_base**((math.log10(ct + 10))**2) # TODO: what should this be set to?
            lr = .95**((math.log10(ct + 10))**2) # TODO: what should this be set to?
            # lr
            r = np.dot(rweights,Fsa)
            s_ = sim.get_next_state(s, a)
            a_opt, idx_a_opt, Qsa_opt, Fsa_opt = select_action_from_policy_exact(Qfsa, s_, 0., start, goal, xd_bins, yd_bins)
            Qfsa[tuple(idxs)] = Qsa + lr*(r + discount*Qsa_opt - Qsa)
            # print("r is: " + str(r))
            # print("Qsa is: " + str(Qsa))
            # print("Qsa_ is: " + str(Qsa_))
            # print("delta is: " + str(delta))
            # input("analyze...")
            s = s_
            if on_policy:
                a = a_opt
                Qsa = Qsa_opt
                Fsa = Fsa_opt
            else:
                a_, idx_a_, Qsa_, Fsa_ = select_action_from_policy_exact(Qfsa, s_, .1, start, goal, xd_bins, yd_bins)
                a = a_
                Qsa = Qsa_
                Fsa = Fsa_
            ct += 1
            # check state for termination conditions:
            termination, termination_result, termination_reason, termination_details = is_state_terminal(s, goal)
            if termination:
                termination_msg = "IRL_iter: " + str(IRL_iter) + "\tCt: " + str(ct) + "\tEpisode: " + str(episode) + "\tIteration: " + str(t) + "\tlr=" + str(lr)
                termination_msg += "\tResult: " + str(termination_result) + "\tReason: " + str(termination_reason)
                termination_msg += "\nState: " + str(termination_details)
            t += 1
            ct += 1
        episode += 1

        dif = np.linalg.norm(_Qfsa - Qfsa)
        print("\n\n***********************************************************************")
        print(termination_msg)
        # input("pausing:")
        print("shape of Qfsa is: " + str(np.shape(Qfsa)))
        print("dif is: " + str(dif))
        # input("pausing after completion of an episode::")
        _Qfsa = np.copy(Qfsa)
    print("dif is: " + str(dif))
    # input("pausing:")

    return Qfsa


    def optimal_action_exact(Qfsa, state, start, goal, xd_bins, yd_bins):
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
        Fsa = np.zeros(k*dparts)
        Fsa = map_features(state, start, goal, xd_bins, yd_bins, action)
        Fsa_list.append(Fsa)
        idxs = Qfsa_map_idx(Fsa, xd_bins, yd_bins)
        Qsa_array[idx] = Qfsa[tuple(idxs)]
        idx += 1
    optimal_idx = np.argmax(Qsa_array)
    optimal_a = global_actions[optimal_idx]
    if plog:
        print("optimal_idx is: " + str(optimal_idx))
        print("optimal_a is: " + str(optimal_a))
        # input("in optimal_action_exact()")
    Qsa = Qsa_array[optimal_idx]
    Fsa = Fsa_list[optimal_idx]
    return optimal_a, optimal_idx, Qsa, Fsa


def optimal_action_wlookahead(pweights, state, start, goal, xd_bins, yd_bins):
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
        Fsa = np.zeros(k*dparts+1)
        Fsa[0:k*dparts] = map_features(state, start, goal, xd_bins, yd_bins, action)
        Fsa[k*dparts] = 1.
        Fsa_list.append(Fsa)
        qsa_cur = np.dot(pweights, Fsa)
        Qsa_array_ = np.zeros(num_actions**2)

        qsa_next = 0.
        idx_ = 0
        for action_ in global_actions:
            state_ = sim.get_next_state(state, action_)
            Fsa_ = np.zeros(k*dparts+1)
            Fsa_[0:k*dparts] = map_features(state_, start, goal, xd_bins, yd_bins, action_)
            Fsa_[k*dparts] = 1.
            # Fsa_list.append(Fsa_)
            Qsa_array_[idx_] = np.dot(pweights, Fsa_)
            idx_ += 1
        qsa_next = max(Qsa_array_)

        Qsa_array[idx] = np.dot(pweights, Fsa) + discount*qsa_next
        idx += 1
    optimal_idx = np.argmax(Qsa_array)
    optimal_a = global_actions[optimal_idx]
    if plog:
        print("optimal_idx is: " + str(optimal_idx))
        print("optimal_a is: " + str(optimal_a))
    Qsa = Qsa_array[optimal_idx]
    Fsa = Fsa_list[optimal_idx]
    return optimal_a, optimal_idx, Qsa, Fsa

def select_action_from_policy_exact(Qfsa, s, epsilon, start, goal, xd_bins, yd_bins):
    u = np.random.uniform(0.,1.)
    if u < (1. - epsilon):
        # print("optimal action being taken")
        optimal_a, idx, Qsa, Fsa = optimal_action_exact(Qfsa, s, start, goal, xd_bins, yd_bins)
        return optimal_a, idx, Qsa, Fsa
    else:
        # print("RANDOM action being taken")
        random_idx = np.random.random_integers(0, (num_actions**2) -1)
        random_a = global_actions[random_idx]
        Fsa = np.zeros(k*dparts)
        Fsa = map_features(s, start, goal, xd_bins, yd_bins, random_a)
        idxs = Qfsa_map_idx(Fsa, xd_bins, yd_bins)
        Qsa = Qfsa[tuple(idxs)]
        return random_a, random_idx,  Qsa, Fsa


# def compute_optimal_policy(reward_weights, start, goal):
#     # randomly sample 1000 states:
#     M = num_rollouts # need to increase this, but set low for debugging purposes TODO
#     # generate 2D array of possible actions
#     actions = global_actions
#     # actions = np.zeros((num_actions**2,2))
#     # idx = 0
#     # for x in range(int(-num_actions/2), int(num_actions/2)):
#     #     for y in range (int(-num_actions/2), int(num_actions/2)):
#     #         actions[idx,0] = x
#     #         actions[idx,1] = y
#     #         idx += 1

#     states = np.zeros((M,state_dim))
#     for i in range(0,M):
#         states[i,0] = np.random.random_integers(0,500)
#         states[i,1] = np.random.random_integers(0,500)
#         states[i,2] = np.random.random_integers(xd_min,xd_max)
#         states[i,3] = np.random.random_integers(yd_min,yd_max)
#         # states[i,4] = np.random.random_integers(xdd_min,xdd_max)
#         # states[i,5] = np.random.random_integers(ydd_min,ydd_max)

#     theta = np.zeros(k_base) # 10 is the number of features of the state, not including the possible actions
#     intercept = 0.
#     dif = np.inf
#     iter_ct = 0
#     while dif > epsilon_RL:
#         print("------------------------------------------------------------------------------\n")
#         print(str(iter_ct) + ":\tdif is: " + str(dif) + "\n")
#         # V_target is our approximation of the "true" value of V*(s_i) for each state s_i = 0....M
#         # this is still an approximation of V*(s_i), but it gets iteratively better, so we can use it as the target values for training the
#         # weights for our general approximator of V*(s) for any state (not just those we sample in i = 1...M) in our regression model
#         V_target = np.zeros(M)
#         for i in range(0,M):
#             if i % 10 == 0:
#                 print("i is: " + str(i))
#             state = states[i]
#             feats_states = np.zeros((M, k_base))
#             q_a = np.zeros((num_actions+1)*2)
#             idx = 0
#             feats_state = map_features(state, start, goal, normalizers, None)
#             feats_states[i,:] = feats_state
#             for action in actions:
#                 state_next = sim.get_next_state(state, action) # note, since the MDP is deterministic we only need to do one sample here...

#                 feats_state = map_features(state, start, goal, normalizers, action)
#                 R_state = np.dot(reward_weights, feats_state)

#                 feats_state_next = map_features(state_next, start, goal, normalizers, None) # passing "None" for "action" indicates that actions should not be included
#                 V_state_next = np.dot(theta, feats_state_next) + intercept

#                 q_a[idx] = R_state + discount*V_state_next
#                 # print("action idx is: " + str(idx))
#                 idx += 1
#             V_target[i] = max(q_a)

#         regr = sk.linear_model.LinearRegression()
#         V_target = np.reshape(V_target, (M,1))
#         # Train the linear regression model
#         regr.fit(feats_states, V_target)


#         new_theta = regr.coef_
#         intercept = regr.intercept_
#         dif = math.fabs(np.linalg.norm(theta) - np.linalg.norm(new_theta))
#         theta = new_theta
#         iter_ct += 1

#     policy = (theta, intercept)
#     return policy

def create_path_from_policy_exact(Qfsa, start, goal, xd_bins, yd_bins, T):
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
        a, idx_a, Qsa, Fsa =  select_action_from_policy_exact(Qfsa, s, 0., start, goal, xd_bins, yd_bins)
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


def optimize_reward_weights(Efeats):
    training_vectors = np.array(Efeats)
    labels = np.zeros(len(Efeats), np.int)
    labels[0] = 1 # set the label for the experts feature expectations to 1...all other labels are set to 0
    clf = sk.svm.SVC(kernel='linear')
    clf.fit(training_vectors, labels)
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

def compute_Efeats_exact(iteration, Qfsa, start, goal, xd_bins, yd_bins, T):
    # plog = True
    plog = False
    flog = False
    if Qfsa is None: # then use random policy:
        Efeats = np.zeros(k*dparts)
        state = np.zeros(state_dim)
        state[_x] = start[0]
        state[_y] = start[1]
        state[_xd] = 0.
        state[_yd] = 0.

        # state[_xd] = np.random.random_integers(-10,10) # note, (-10,10) is pretty arbitrary right now...
        # state[_yd] = np.random.random_integers(-10,10) # note, (-10,10) is pretty arbitrary right now...

        # for t in range(0,T):
        termination = False
        t = 0
        while not termination:
            # action = np.random.random_integers(-7, 7, 2)
            action = global_actions[np.random.random_integers(0, (num_actions**2) - 1)] #TODO: make sure this works...
            # print("rando action is: " + str(action))
            features = map_features(state, start, goal, xd_bins, yd_bins, action)
            Efeats += (math.pow(discount, t) * features) # note that these are both vectors (i.e. 1d ndarrays), not scalars
            # determine next state based on current state and selected (random in this case) action
            state = sim.get_next_state(state, action)
            termination, termination_result, termination_reason, termination_details = is_state_terminal(state, goal)
            t += 1
        Efeats_norm = np.linalg.norm(Efeats)
        Efeats = Efeats / Efeats_norm
        return Efeats, t
    else:
        Efeats = np.zeros(k*dparts)
        for m in range(0,num_rollouts):
            Efeats_m = np.zeros(k*dparts)
            s = np.zeros(state_dim)
            s[_x] = start[0] + np.random.random_integers(-10,10)
            s[_y] = start[1] + np.random.random_integers(-10,10)
            # s[_x] = np.random.random_integers(x_min, x_max)
            # s[_y] = np.random.random_integers(y_min, y_max)
            s[_xd] = 0.
            s[_yd] = 0.

            # actions = global_actions
            # for t in range(0,T):
            termination = False
            t = 0
            # QQQ stuck in this while loop...something is going wrong...need to debug...
            while not termination and t < max_iters_per_episode:
                a, idx_a, Qsa, Fsa = select_action_from_policy_exact(Qfsa, s, 0., start, goal, xd_bins, yd_bins)

                # max_reward = 0.
                # V_max = 0.
                # idx_max = 0
                # for a in range(0, len(actions)):
                #     action = actions[a]
                #     state_next = sim.get_next_state(state, action) # note, since the MDP is deterministic we only need to do one sample here...
                #     # feats_state_next = map_features(state_next, start, goal, normalizers, None) # "None" for "action" indicates actions shouldn't be included

                #     V_state_next = np.dot(pweights, state_next) + intercept
                #     if V_state_next > V_max:
                #         V_max = V_state_next
                #         idx_max = a

                # selected_action = actions[idx_max] # select the best action to take at that time step
                if plog:
                    print("selected action is: " + str(a))

                # features = map_features(state, start, goal, normalizers, selected_action)
                features = Fsa
                Efeats_m += (math.pow(discount, t) * features) # note that these are both vectors (i.e. 1d ndarrays), not scalars
                # determine next state based on current state and selected (random in this case) action
                s = sim.get_next_state(s,a)
                if plog:
                    print("\tt: " + str(t) + "\tnext state is: " + str(s))
                termination, termination_result, termination_reason, termination_details = is_state_terminal(s, goal)
                t += 1
            # Efeats_m_norm = np.linalg.norm(Efeats_m)
            # Efeats_m = Efeats_m / Efeats_m_norm
            Efeats += Efeats_m
        Efeats = Efeats / num_rollouts
        Efeats_norm = np.linalg.norm(Efeats)
        if Efeats_norm > 0:
            Efeats = Efeats / Efeats_norm
        # print("Efeats is: ")
        # print(Efeats)
        # print("Efeats_norm is: ")
        # print(Efeats_norm)
        return Efeats, t


def detect_bins(demos):
    xds = list()
    yds = list()
    for demo in demos:
        for t in range(0, len(demo)):
            xds.append(math.fabs(demo[t,_xd]))
            yds.append(math.fabs(demo[t,_yd]))
    xd_mean = stats.mean(xds)
    yd_mean = stats.mean(yds)
    xd_stdev = stats.stdev(xds, xd_mean)
    yd_stdev = stats.stdev(yds, yd_mean)

    last_split = xd_mean + 2.*xd_stdev
    bin_width = last_split / float(num_xd_bins - 1)
    xd_splits = list()
    for i in range(0, num_xd_bins):
        xd_splits.append(0 + i*bin_width)
    xd_bins = np.zeros((num_xd_bins, 2))
    for i in range(0, num_xd_bins):
        if i < num_xd_bins-1:
            xd_bins[i,0] = xd_splits[i]
            xd_bins[i,1] = xd_splits[i+1]
        else:
            xd_bins[i,0] = xd_splits[i]
            xd_bins[i,1] = np.inf

    last_split = yd_mean + 2.*yd_stdev
    bin_width = last_split / float(num_yd_bins - 1)
    yd_splits = list()
    for i in range(0, num_yd_bins):
        yd_splits.append(0 + i*bin_width)
    yd_bins = np.zeros((num_yd_bins, 2))
    for i in range(0, num_yd_bins):
        if i < num_yd_bins-1:
            yd_bins[i,0] = yd_splits[i]
            yd_bins[i,1] = yd_splits[i+1]
        else:
            yd_bins[i,0] = yd_splits[i]
            yd_bins[i,1] = np.inf

    print("xd_bins is: " + str(np.shape(xd_bins)))
    print(xd_bins)
    print("yd_bins is: " + str(np.shape(yd_bins)))
    print(yd_bins)
    input("check and make sure that xd_bins and yd_bins looks right...")
    return xd_bins, yd_bins



