import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import time
import importlib
import os
from env import gridworld, layout_original, layout_nowalls, layout_original_walled 
from helper import *
from agent import Agent


SIZE = 11
LAYOUT = layout_original_walled
SHOW = False
EPISODES = 20001
SHIELD_ON = True
N_ACTIONS = 5 # N_ACTIONS must be 5 or 9 (including standing still)
MOVE_PENALTY = -1
ENEMY_PENALTY = -300
TARGET_REWARD = 25
WALL_PENALTY = 0

SHOW_EVERY = 1000
SAVE_INTERVAL = 1#EPISODES - 1

EPSILON_START=0.3
EPSILON_END=0.1 #0.02 # 0.1
EPSILON_DECAY=EPISODES

# LEARNING_RATE = 5e-4 #0.1
LEARNING_RATE_START = 0.2
LEARNING_RATE_END = 0.01
LEARNING_RATE_DECAY = EPISODES - 1

DISCOUNT = 0.95

start_q_table = None # insert qtable filename if available/saved
SAVE_Q_TABLE =  False
SAVE_RESULTS = False
PLOT = True

# random.seed(34095)
# np.random.seed(34095)

# checks
assert N_ACTIONS == 5 or N_ACTIONS == 9, "N_ACTIONS can only be 5 or 9"

# import shield
if SHIELD_ON:
    try:
        mod_name = f"9x9_3_{str(N_ACTIONS - 1)}directions"
        Shield = importlib.import_module(mod_name).Shield
        shield = Shield()
    except ImportError as e:
        print("Could not find shield.")

# qlearning algo
q_table = generate_qtable(start_q_table, SIZE, N_ACTIONS)

episode_rewards = [0]
walls = gridworld(LAYOUT, SIZE).walls
for episode in range(EPISODES):
    # if episode % SAVE_INTERVAL == 0 and episode > 1:
    #     save_results(q_table, episode_rewards, SHIELD_ON, EPISODES, N_ACTIONS, SAVE_RESULTS, SAVE_Q_TABLE)

    epsilon = np.interp(episode, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    lr = np.interp(episode, [0, LEARNING_RATE_DECAY], [LEARNING_RATE_START, LEARNING_RATE_END])

    places_no_walls = no_walls(SIZE, walls)
    player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True) #x=4, y=7,
    target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True) #x=6, y=0, 
    # enemy = Agent(places_no_walls, random_init=True)

    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}, epsilon: {epsilon}, lr: {lr}, mean reward: {np.mean(episode_rewards[-SHOW_EVERY:])}")

    # steps in episode
    episode_reward = 0
    for i in range(100):
        # obs = (player-target, player-enemy)
        t0_obs = (player-target)

        rnd = np.random.random()
        if SHIELD_ON:
            t0_action, t1_action, t1_obs = shielded_actions(rnd, epsilon, q_table, t0_obs, N_ACTIONS, player, walls, target, shield)
        else:
            t0_action = unshielded_action(rnd, epsilon, q_table, t0_obs, N_ACTIONS)

        reward, done = check_reward(player=player, target=target, action=t0_action, walls=walls, TARGET_REWARD=TARGET_REWARD,
                                WALL_PENALTY=WALL_PENALTY, MOVE_PENALTY=MOVE_PENALTY)
        episode_reward += reward
        # print(reward)
        # live calculation new qvalue (certain for timestep 1 (t1), not yet certain for t2)
        current_q = q_table[t0_obs][t0_action]
        max_future_q = q_table[t1_obs][t1_action]
        new_q = (1 - lr) * current_q + lr * (reward + DISCOUNT * max_future_q)
        q_table[t0_obs][t0_action] = new_q

        # perform action
        player.action(t0_action)

        # render visualization
        if SHOW:
            env = gridworld(layout=LAYOUT, size=SIZE)
            env.render(player, target, step=i, reward=reward)

        # if target/wall/enemy is hit reset the game
        if done:
            break
        
    # save reward
    episode_rewards.append(episode_reward)

# plot directly
if PLOT:
    plot(episode_rewards, SHOW_EVERY)
