
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

import os
import os.path
import gym
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import time
from drpo import DRTRPOAgent 
from PowerDynSimEnvDef_v5 import PowerDynSimEnv


java_port = 25332
jar_file = "/lib/RLGCJavaServer0.89.jar"

a = os.path.abspath(os.path.dirname(__file__))
# This is to fix the issue of "ModuleNotFoundError" below
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

folder_dir = a[:-7]
jar_path = folder_dir + jar_file

case_files_array = []

case_files_array.append(folder_dir +'/testData/Kundur-2area/kunder_2area_ver30.raw')
case_files_array.append(folder_dir+'/testData/Kundur-2area/kunder_2area.dyr')

dyn_config_file = folder_dir+'/testData/Kundur-2area/json/kundur2area_dyn_config.json'

rl_config_file = folder_dir+'/testData/Kundur-2area/json/kundur2area_RL_config_multiStepObsv.json'


env = PowerDynSimEnv(case_files_array,dyn_config_file,rl_config_file,jar_path,java_port)

# Check agent class for initialization parameters and initialize agent
gamma = 0.99
lr = 1e-3
    
agent = DRTRPOAgent(env, gamma, lr)

# Define training parameters
max_episodes = 50
max_steps = 1000
total_adv_diff = 0

episode_rewards = []
run_time = []
start_time = time.time()
for episode in range(max_episodes):
    if episode == 0:
        first_state = env.reset()
    else:
        first_state = state
    state_adv = []
    total_value_loss = 0
    
    episode_reward = 0
    # loop through the first action
    for i in range(env.action_space.n):
        env.reset()
        state = first_state
        action = i
        trajectory = []
        
        for step in range(max_steps):
            if step != 0:
                action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            episode_reward += reward  
            if done or step == max_steps:
                break
            state = next_state
            
        adv, value_loss = agent.compute_adv_mc(trajectory)
        state_adv.append(adv[0])
        total_value_loss += value_loss
    
    avg_episode_reward = episode_reward/env.action_space.n
        
    total_adv_diff += max(abs(state_adv[1] - state_adv[0]), abs(state_adv[2] - state_adv[0]), abs(state_adv[2] - state_adv[1]))
    beta = total_adv_diff/episode
    beta += 0.1
    policy_loss = agent.compute_policy_loss_wass(first_state, state_adv, beta)

    agent.update(value_loss, policy_loss)
    elapse = time.time() - start_time
    run_time.append(elapse)
    
    episode_rewards.append(avg_episode_reward)
    print("Episode " + str(episode) + ": " + str(avg_episode_reward))
    print("Timesteps in the episode: " + str(step))