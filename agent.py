import os.path as path

import random
from collections import namedtuple, deque

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)
random.seed(0)

transition_tuple = namedtuple('transition_tuple', ['state', 'action', 'state_next', 'reward', 'done'])

class Agent:
    def __init__(self, state_size, action_size, 
        gamma = 0.99, learning_rate = 0.001, learn_every = 5, 
        soft_update_rate = 0.005, soft_update_every = 50,
        epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.01,
        training_batch_size = 32, buffer_size = 50000, 
        train_file = 'training.sav'):

        self.train_file = train_file

        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = gamma
        self.learn_every = learn_every
        self.training_batch_size = training_batch_size
        self.soft_update_rate = soft_update_rate
        self.soft_update_every = soft_update_every

        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.transitions_buffer = deque(maxlen = buffer_size)
        self.nn_q = nn.Sequential(
            nn.Linear(state_size, 80), nn.ReLU(),
            nn.Linear(80, 48), nn.ReLU(),
            nn.Linear(48, action_size)
        )
        self.nn_q_target = nn.Sequential(
            nn.Linear(state_size, 80), nn.ReLU(),
            nn.Linear(80, 48), nn.ReLU(),
            nn.Linear(48, action_size)
        )
        self.optimizer = optim.Adam(self.nn_q.parameters(), lr=learning_rate)

        self.training_counter = 0
        self.soft_update_counter = 0
        self.training_mod = 0
        self.soft_update_mod = 0
        
        self.episode_counter = 0
        self.rewards_window = deque(maxlen=100)
        self.average_reward = 0

    ##################################################
    #Learning cycle functions
    ##################################################
    
    def select_action(self, state):
        #Ep-Greedy selection
        self.nn_q.eval()
        with torch.no_grad():
            state = torch.tensor(state.reshape([1, self.state_size]), dtype=torch.float32)
            action_values = self.nn_q(state)
        if random.random() > self.epsilon:
            return np.argmax(action_values.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def add_transition(self, state, action, state_next, reward, done):
        #Save transition in buffer
        self.transitions_buffer.append(
            transition_tuple(state, action, state_next, reward, done) )

    def trigger_learning(self):
        self.training_mod = (self.training_mod + 1) % self.learn_every
        if( self.training_mod == 0 and
                len(self.transitions_buffer) >= self.training_batch_size ):
            #Get sample of transitions (experiences)
            batch = random.sample(self.transitions_buffer, self.training_batch_size)

            #Transpose sample (from list of tuples to tupple of lists)
            batch = transition_tuple(*zip(*batch))

            #Get next states and index of non final states
            states_next = torch.tensor(batch.state_next, dtype=torch.float32)
            non_final_index = [ i for i in range(self.training_batch_size) if not batch.done[i] ]
            
            #Evaluate target neural net Q in non final states
            self.nn_q_target.eval()
            with torch.no_grad():
                q_next = torch.zeros((self.training_batch_size, self.action_size), dtype=torch.float32)
                q_next[non_final_index, :] = self.nn_q_target(states_next[non_final_index, :]).detach()
            q_next = torch.max(q_next, dim=1)[0].unsqueeze(1)

            #Compute target state-action values
            rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
            q_target = rewards + self.gamma*q_next

            #Compute predicted state-action values
            states = torch.tensor(batch.state, dtype=torch.float32)
            actions = torch.tensor(batch.action).long().unsqueeze(1)
            self.nn_q.train()
            q_expected = self.nn_q(states).gather(1, actions)

            #Training
            self.optimizer.zero_grad()
            self.nn_q.zero_grad()
            loss = F.mse_loss(q_expected, q_target)
            loss.backward()
            self.optimizer.step()
            
            self.q_target_soft_update(self.soft_update_rate)
            self.training_counter += 1

    def q_target_soft_update(self, factor):
        self.soft_update_mod = (self.soft_update_mod + 1) % self.soft_update_every
        if( self.soft_update_mod == 0 ): 
            #Copy parameters from neural net Q to target neural net Q
            for param, target_param in zip(self.nn_q.parameters(), self.nn_q_target.parameters()):
                target_param.data.copy_(factor*param.data + (1.0-factor)*target_param.data)
            self.soft_update_counter += 1

    def decrease_epsilon(self, episode):
        #self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)
        self.epsilon = self.epsilon_min + \
            (self.epsilon_max-self.epsilon_min) * \
                math.exp(-1.0*episode/self.epsilon_decay)

    def finish_episode(self, episode_reward):
        self.episode_counter += 1
        self.rewards_window.append(episode_reward)
        if( len(self.rewards_window)>0 ):
            self.average_reward = np.average( self.rewards_window )

    ##################################################
    #Progress saving functions
    ##################################################
    
    def load_training_if_exists(self):
        if path.isfile(self.train_file):
            loaded = torch.load(self.train_file)
            self.nn_q.load_state_dict(loaded['nn_q_state_dict'])
            self.optimizer.load_state_dict(loaded['optim_state_dict'])
            self.nn_q_target.load_state_dict(loaded['nn_q_target_state_dict'])
            self.transitions_buffer = loaded['transitions_buffer']
            
            self.epsilon = loaded['epsilon']
            self.training_counter = loaded['training_counter']
            self.soft_update_counter = loaded['soft_update_counter']
            self.training_mod = loaded['training_mod']
            self.soft_update_mod = loaded['soft_update_mod']

            self.episode_counter = loaded['episode_counter']
            self.rewards_window = loaded['rewards_window']
            self.average_reward = loaded['average_reward']
            
    def save_training(self):
        training = {
            'nn_q_state_dict': self.nn_q.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'nn_q_target_state_dict': self.nn_q_target.state_dict(),
            'transitions_buffer': self.transitions_buffer,
            
            'epsilon': self.epsilon,
            'training_counter': self.training_counter,
            'soft_update_counter': self.soft_update_counter,
            'training_mod': self.training_mod,
            'soft_update_mod': self.soft_update_mod,
            
            'episode_counter': self.episode_counter,
            'rewards_window': self.rewards_window,
            'average_reward': self.average_reward
        }
        torch.save(training, self.train_file)
