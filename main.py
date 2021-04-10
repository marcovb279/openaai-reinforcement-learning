import sys
import os
import os.path as path
from collections import namedtuple, deque
import csv

import keyboard
import numpy as np

import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym import spaces
from agent import Agent

##################################################
#Create env
##################################################
envName = 'LunarLander-v2'
envName = 'CartPole-v1'
env = gym.make(envName)
env.seed(0)

##################################################
#Create folder for data
##################################################
training_folder = './training/'
if not os.path.exists(training_folder):
    os.makedirs(training_folder)
    
##################################################
#Configuration parameters
##################################################
EPISODES = 10000
EPISODE_MAX_STEPS = env.spec.max_episode_steps
VIDEO_EVERY = 50

GAMMA = 0.99
LEARNING_RATE = 0.002
LEARN_EVERY = 5
SOFT_UPDATE_RATE = 0.002
SOFT_UPDATE_EVERY = 1
TRAINING_BATCH_SIZE = 100
BUFFER_LIMIT_SIZE = 30000

EPSILON_INITIAL = 1.0
EPSILON_DECAY = 210
EPSILON_MIN = 0.01

LEARN_THRESHOLD = env.spec.reward_threshold
PRINT_MOD = 10

##################################################
#Create agent
##################################################
agent = Agent(env.observation_space.shape[0], env.action_space.n, 
            GAMMA, LEARNING_RATE, LEARN_EVERY,
            SOFT_UPDATE_RATE, SOFT_UPDATE_EVERY,
            EPSILON_INITIAL, EPSILON_DECAY, EPSILON_MIN,
            TRAINING_BATCH_SIZE, BUFFER_LIMIT_SIZE, 
            training_folder + 'Train-'+envName+'.sav')
agent.load_training_if_exists()

##################################################
#Register hotkeys listeners
##################################################
#Listener for quit
exit_flag = False
def enable_exit(): 
    global exit_flag
    exit_flag = True
keyboard.add_hotkey('shift+escape', enable_exit)
#Listener for render
render_flag = False
def toggle_render(): 
    global render_flag
    render_flag = not render_flag
keyboard.add_hotkey('shift+f1', toggle_render)
#Listener for recording
record_flag = False
def enable_recording(): 
    global record_flag
    record_flag = True
keyboard.add_hotkey('shift+f2', enable_recording)

##################################################
#Train stats file csv
##################################################
csv_file = open(training_folder + envName + '.csv', mode='a+', newline='')
csv_writer  = csv.writer(csv_file, delimiter=',')

##################################################
#Start episodes loop
##################################################
learning_flag = agent.average_reward < LEARN_THRESHOLD
record_flag = agent.average_reward < LEARN_THRESHOLD
for episode in range(agent.episode_counter+1, EPISODES):
    state = env.reset()
    cumulative_reward = 0
    steps = 0
    done = False

    #Create video recorder if hotkey pressed
    if not record_flag: record_flag = episode%VIDEO_EVERY == 0
    is_recording = False
    if record_flag:
        record_flag = False
        is_recording = True
        videoRec = VideoRecorder(env, training_folder +'Vid-' + envName \
            + '-Ep-' + str(episode) + '.mp4', metadata=None)

    #Run episode
    while not done and steps <= EPISODE_MAX_STEPS:        
        #Render and/or add frame to video
        if is_recording: videoRec.capture_frame()
        elif render_flag: env.render()
        else: env.close()

        #Quit
        if exit_flag:
            print('\n' + 'Exit...')
            if learning_flag: agent.save_training()
            if is_recording: videoRec.close()
            env.close()
            csv_file.close()
            quit(0)
        
        #Predict action and save transition
        action = agent.select_action(state)
        state_next, reward, done, info = env.step(action)
        #Learn
        if learning_flag: 
            agent.add_transition(state, action, state_next, reward, done)
            agent.trigger_learning()
        
        #Update for next iteration
        state = state_next
        cumulative_reward += reward
        steps += 1

    #Close video recorder after episode
    if is_recording: videoRec.close()

    #Update learning stats
    if learning_flag:
        agent.finish_episode(cumulative_reward)
        agent.decrease_epsilon(episode)
        csv_writer.writerow([episode, steps, agent.training_counter, 
            cumulative_reward, agent.average_reward, learning_flag])
        csv_file.flush()
        if( agent.average_reward >= LEARN_THRESHOLD ):
            agent.save_training()
            learning_flag = False
            record_flag = True

    #Show stats
    message =  ('Episode: {0:6d},  '
                'Steps: {1:5d},  '
                'Reward: {2:8.2f},  ' 
                'Average: {3:8.2f},  '
                'Solved:  {4},  '
                'Training: {5:7d},  '
                'Eps: {6:4.2f}').format(
                    episode, 
                    steps,
                    cumulative_reward,
                    agent.average_reward,
                    not learning_flag,
                    agent.training_counter,
                    agent.epsilon)
    print('\r' + message, end='')
    if( episode%PRINT_MOD==0 ): print('\r' + message)

if learning_flag: agent.save_training()
env.close()