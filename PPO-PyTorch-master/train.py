import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import roboschool
from gurobipy import GRB,Model,quicksum

from PPO import PPO
from dataset import QAPLIB

def LNS4instances(model,D,F,action,sol,verbose = False):    
    prob_index = [i for i in range(D.shape[1])]

    next_sols = []
    next_objs = []

    if action.shape[1] <= 1:
        # f,d,sol = f.cpu().numpy(), d.cpu().numpy(),sol.cpu().numpy()
        obj = cal_obj_np(sol,D,F)
        
        return sol,obj
    
    # f,d,sol = f.cpu().numpy(), d.cpu().numpy(),sol.cpu().numpy()

    index = sorted(action[0].tolist())
    out_index = np.setdiff1d(np.array(prob_index),np.array(index)).tolist()
    sol,obj = LNS4RL(model.copy(),D,F,sol,index,out_index,0)
    # next_sols.append(torch.tensor(sol_))
    # next_objs.append(torch.tensor([obj_]))
        
    return  sol, obj

def LNS4RL(model,D,F,sol,index, out_index, start_time):
    N = len(index)

    assert N > 0 ,'N 必须大于0'

    per_sol = form_per_np(sol)
    per_assigned = per_sol[out_index]
    sub_loc = sorted([j for i,j in sol if i in index])

    x = model.addMVar(shape=(N,N),vtype= GRB.BINARY,name='x')
    
    if N == 20:
        objective = (F*(x@D@x.T)).sum()
        model.setObjective(objective,GRB.MINIMIZE)

        model.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
        model.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N)); 

        model.optimize()

        sol = solution(model)
        obj = model.objVal

        return sol, obj
    else:
        sub_D = D[sub_loc][:,sub_loc]
        sub_F = F[index][:,index]
        sub_left_D = D[sub_loc] @ per_assigned.T
        sub_right_D = per_assigned @ D[:,sub_loc]
        sub_left_F = F[index][:,out_index]
        sub_right_F = F[out_index][:,index]


        objective = (sub_F*(x@sub_D@x.T)).sum() # 二次项
        objective += (sub_left_F*(x@sub_left_D)).sum() 
        objective += (sub_right_F*(sub_right_D@x.T)).sum()#一次项
        model.setObjective(objective,GRB.MINIMIZE)

        model.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
        model.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N)); 

        model.optimize()
        
        sub_sol = solution(model)
        sol_new = sub2whole(sub_sol,sol,index)
        obj = cal_obj_np(sol_new,D,F)

        # import pdb; pdb.set_trace()
        
        # gurobi_interm_obj.append(obj)
        # gurobi_interm_time.append(time.time()-start_time)

        return sol_new, obj

def form_per(sol,device):
    N = len(sol)
    per = torch.zeros((N,N)).to(device)
    
    for i,j in sol:
        per[i,j] = 1

    return per

def form_per_np(sol):
    N = len(sol)
    per = np.zeros((N,N))
    
    for i,j in sol:
        per[i,j] = 1

    return per

def cal_obj(sol,D,F,device):
    per = form_per(sol,device)
    obj = torch.sum(F*(per@D@per.T))  
    return obj

def cal_obj_np(sol,D,F):
    per = form_per_np(sol)
    obj = np.sum(F*(per@D@per.T))  
    return obj

def sub2whole(sub_sol,sol,index):
    sol_out = sol[:]
    sub_prob_size = len(sub_sol)
    unassigned_loc = sorted([loc for idx, loc in sol_out if idx in index])
    unassigned_flow = index

    for i in range(sub_prob_size):
        sol_out[unassigned_flow[i]][1] = unassigned_loc[sub_sol[i][1]]
    
    return sol_out
             
def solution(model):
    sol = []
    for v in model.getVars():
        if v.x == 1:
            sol.append(eval(v.VarName[1:]))
    return sol

def mycallback(model, where):
    # 如果是在找到一个新的解决方案时
    if where == GRB.Callback.MIPSOL:
        # 获取当前解的目标值
        objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        gurobi_interm_obj.append(objval)
        gurobi_interm_time.append(runtime)

def initial_sol(F,D):
    '''
        We offer mutiple initial methods including:
        1) random initialize 
        2) learnable network initialize
    '''
    
    N = F.shape[0]
    
    ####random initialize
    prob_index = [i for i in range(N)]
    # init_loc = random.sample(prob_index,len(prob_index))
    # sol = [[i,j] for i,j in enumerate(init_loc)]
    sol = [[i,i] for i in range(N)]

    obj = cal_obj_np(sol,D,F)

    ####learnable network initialize(TODO)

    return sol,obj

def featurize(F,D,cur_sol):
    n,n = F.shape
    # import pdb;pdb.set_trace()
    
    features = np.zeros((n,3*n))

    sol = form_per_np(cur_sol)
    loc_fea = np.matmul(sol,D)
    features = np.concatenate([F,sol,loc_fea] , axis= -1)

    return features 



################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "erdos20"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 2                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 1        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 10               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 1.            # discount factor

    lr_actor = 0.00003 #0.0003       # learning rate for actor network
    lr_critic = 0.0001       # learning rate for critic network

    local_size = 12
    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # print("training environment name : " + env_name)

    writer = SummaryWriter('runs/{}_PPO_{}_lr_actor_{}_lr_cirtic_{}_clips_{}_gamma_{}_K_epochs_{}_local_{}_seed_{}'.format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name,lr_actor,lr_critic,eps_clip,gamma,K_epochs,local_size,random_seed))

    # env = gym.make(env_name)
    train_set = QAPLIB('train','erdos')
    F,D,per,sol,name, opt_obj = train_set.get_pair(0)

    # state space dimension
    # state_dim = env.observation_space.shape[0]
    state_dim = 60

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 20

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, local_size ,action_std)
    
    model = Model('QAP')
    model.Params.TimeLimit = 5    
    model.Params.OutputFlag = 0

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 1

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        # state = env.reset()
        init_sol, init_obj = initial_sol(F,D)
        state = featurize(F,D,init_sol)

        current_ep_reward = 0
        cur_sol = init_sol
        cur_obj = init_obj

        for t in range(1, max_ep_len+1):
            # select action with policy
            action, action_prob= ppo_agent.select_action(state)
            indices = np.array([np.where(action == 1)[1]])

            # import pdb; pdb.set_trace()

            # state, reward, done, _ = env.step(action)
            next_sol,next_obj = LNS4instances(model,F,D,indices,cur_sol)
            # saving reward and is_terminals

            reward = (cur_obj - next_obj)/1000
            state = featurize(F,D,next_sol)

            cur_obj = next_obj
            # cur_sol = next_sol    

            while t == max_ep_len+1:
                done = True
            else:
                done = False

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # import pdb; pdb.set_trace()
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Objective : {}".format(i_episode, time_step, print_avg_reward,next_obj))
                print(action_prob)
                print(action)
                writer.add_scalar('reward/train', print_running_reward, i_episode)
                
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path+str(datetime.now()))
                ppo_agent.save(checkpoint_path + str(datetime.now()))
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    # env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
