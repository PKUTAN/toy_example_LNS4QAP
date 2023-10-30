
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess, NormalActionNoise
from util import *

from copy import deepcopy
# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, instance_batchsize, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.prob_size = nb_states//3
        self.instance_batch_size = instance_batchsize
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize,prob_size=self.prob_size, window_length=args.window_length)
        self.random_process = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(0.01) * np.ones(nb_actions))

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        self.actor_loss = 0
        self.critic_loss = 0
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        state_batch = to_tensor(state_batch)
        action_batch = to_tensor(action_batch)
        reward_batch = to_tensor(reward_batch)
        next_state_batch = to_tensor(next_state_batch)
        terminal_batch = to_tensor(terminal_batch.astype(np.float))

        # Prepare for the target q batch
        next_action = self.actor_target(next_state_batch)
        next_q_values = self.critic_target([next_state_batch,next_action.detach()])
        # with torch.no_grad():
        #     next_q_values = self.critic_target([
        #         to_tensor(next_state_batch),
        #         self.actor_target(to_tensor(next_state_batch)),
        #     ])

        # import pdb; pdb.set_trace()

        target_q_batch = reward_batch + \
            self.discount*terminal_batch*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([state_batch, action_batch])
        
        value_loss = criterion(q_batch, target_q_batch)
        self.critic_loss = value_loss
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        
        
        action_batch_prob = torch.abs(self.actor(state_batch)-(1-action_batch))
        # import pdb; pdb.set_trace()

        action_batch_prob = torch.mean(torch.log(action_batch_prob),dim=-1) ####训练不稳定

        policy_loss = -   self.critic([
            state_batch,
            self.actor(state_batch)
        ])

        policy_loss = policy_loss.mean()
        self.actor_loss = policy_loss
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            bts = r_t.shape[0]
            # import pdb; pdb.set_trace()
            for b in range(bts):
                self.memory.append(np.array([self.s_t[b]]), np.array([self.a_t[b]]), np.array([r_t[b]]), done)
            self.s_t = s_t1

    def random_action(self,local_size):
        actions = []
        for i in range(self.instance_batch_size):
            actions.append(np.array([np.random.uniform(0,1.,self.nb_actions) for _ in range(self.prob_size)]))
        
        actions = np.stack(actions,axis=0).reshape(self.instance_batch_size,-1)

        actions_prob = actions
        indices = torch.topk(torch.from_numpy(actions),local_size)[1].numpy()
        actions = np.zeros_like(actions)
        actions[:,indices] = 1.
        self.a_t = actions
        
        return actions_prob, indices
    
    def select_action(self, s_t,local_size,prob_size , decay_epsilon=True):
        actions = to_numpy(
            self.actor(s_t)
        )
        # actions += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        # import pdb; pdb.set_trace()

        actions_prob = deepcopy(actions)
       
        actions = np.clip(actions,0.1,0.9)
        actions = np.random.binomial(1, actions) 
        # actions = np.round(actions) 
        # actions =np.where(actions > 0.5, actions, 0.)  
        # actions =np.where(actions == 0., actions, 1.)  
        indices = np.array([np.where(actions == 1)[1]])

        # import pdb; pdb.set_trace()
        # if indices.shape[1] > local_size  or indices.shape[1] < local_size -2:
        #     # import pdb; pdb.set_trace()
        #     if indices.shape[1] > local_size :
        #         sort_index = torch.topk(torch.from_numpy(-actions_prob),prob_size)[1].numpy()
        #         eraze = indices.shape[1]-local_size 
        #         for i in range(prob_size):
        #             if eraze == 0:
        #                 break
        #             if actions[0][sort_index[0][i]] == 1:
        #                 actions[0][sort_index[0][i]] = 0.
        #                 eraze -= 1
        #         indices = np.array([np.where(actions == 1)[1]])
        #     elif indices.shape[1] < local_size -2:
        #         sort_index = torch.topk(torch.from_numpy(actions_prob),prob_size)[1].numpy()
        #         eraze = local_size-2 - indices.shape[1]
        #         for i in range(prob_size):
        #             if eraze == 0:
        #                 break
        #             if actions[0][sort_index[0][i]] == 0:
        #                 actions[0][sort_index[0][i]] = 1.
        #                 eraze -= 1
        #         indices = np.array([np.where(actions == 1)[1]])

        # import pdb; pdb.set_trace()
        # print(indices.shape)
        # print(indices.shape)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = actions
        return actions_prob, indices

    def reset(self, obs):
        self.s_t = obs
        # self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
