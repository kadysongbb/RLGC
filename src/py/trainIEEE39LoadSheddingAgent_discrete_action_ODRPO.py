
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

import os
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
import json
from drpo import DRTRPOAgent 
from PowerDynSimEnvDef_v5 import PowerDynSimEnv


np.random.seed(19)

# config the RLGC Java Sever
java_port = 25002
jar_file = '/lib/RLGCJavaServer0.87.jar'

a = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

repo_path = a[:-7]
jar_path = repo_path + jar_file

case_files_array =[]

case_files_array.append(repo_path + '/testData/IEEE39/IEEE39bus_multiloads_xfmr4_smallX_v30.raw')
case_files_array.append(repo_path + '/testData/IEEE39/IEEE39bus_3AC.dyr')

# configuration files for dynamic simulation and RL
dyn_config_file = repo_path + '/testData/IEEE39/json/IEEE39_dyn_config.json'
rl_config_file = repo_path + '/testData/IEEE39/json/IEEE39_RL_loadShedding_3motor_2levels.json'


storedData = "./storedData"

savedModel= "./trainedModels"
model_name = "IEEE39_multistep_obs11_randftd3_randbus3_3motor2action_prenull"


env = PowerDynSimEnv(case_files_array,dyn_config_file,rl_config_file, jar_path, java_port)

# Check agent class for initialization parameters and initialize agent
gamma = 0.99
lr = 1e-3
    
agent = DRTRPOAgent(env, gamma, lr)

# Define training parameters
max_episodes = 20
max_steps = 1000
total_adv_diff = 0

episode_rewards = []

results_dict = {
    'train_rewards': [],
    'eval_rewards': [],
    'actor_losses': [],
    'value_losses': [],
    'critic_losses': []
}
total_timesteps = 0

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
    total_timesteps += step * env.action_space.n

    results_dict['train_rewards'].append(
        (total_timesteps, avg_episode_reward))

    with open('results.txt', 'w') as file:
        file.write(json.dumps(results_dict))

    agent.update(value_loss, policy_loss)

    episode_rewards.append(avg_episode_reward)
    print("Episode " + str(episode) + ": " + str(avg_episode_reward))