import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import time
import importlib
import os
from env import gridworld, layout_original, layout_original_walled, layout_nowalls, layout_example_1, layout_example_2, layout_example_3, layout_original_walled_big
from helper import *
from agent import Agent


def run_experiment(
	SIZE = 11,
	LAYOUT = layout_original_walled,
	SHOW = False,
	EPISODES = 5000,
	SHIELD_ON = False,
	CQL = False,
	N_ACTIONS = 5, # N_ACTIONS must be 5 or 9 (including standing still)
	DYNAMIC = False,
	DISCOUNT = 0.95,
	MOVE_PENALTY = -1,
	# ENEMY_PENALTY = -300,
	TARGET_REWARD = 25,
	WALL_PENALTY = -5,

	RS = False,
	RS_PENALTY = -1,

	SHOW_EVERY = 100,

	EPSILON_START=0.3,
	EPSILON_END=0.01, #0.02 # 0.1

	# LEARNING_RATE = 5e-4 #0.1
	LEARNING_RATE_START = 0.2,
	LEARNING_RATE_END = 0.05,

	start_q_table = 'qtable_5000_5Actions_Shielded', # insert qtable filename if available/saved
	SAVE_Q_TABLE =  True,
	SAVE_RESULTS = True,
	):
	
	SAVE_INTERVAL = EPISODES - 1
	EPSILON_DECAY = EPISODES
	LEARNING_RATE_DECAY = EPISODES - 1
	# random.seed(34095)
	# np.random.seed(34095)

	# checks
	assert N_ACTIONS == 5 or N_ACTIONS == 9, "N_ACTIONS can only be 5 or 9"

	# if trained q table used, only do exploitation and show
	if start_q_table:
		EPSILON_START=0
		EPSILON_END=0
		SHOW=True
	# print settings (for notebook)
	print(f"Agent summary --> Size: {SIZE}, Episodes: {EPISODES}, SHIELD_ON: {SHIELD_ON}, CQL: {CQL}, N_ACTION: {N_ACTIONS}, DISCOUNT: {DISCOUNT},\
	MOVE_PENALTY: {MOVE_PENALTY}, TARGET_REWARD: {TARGET_REWARD}, WALL_PENALTY: {WALL_PENALTY}, RS: {RS}, RS_PENALTY: {RS_PENALTY}")
	
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
		if episode % SAVE_INTERVAL == 0 and episode > 1:
			save_results(q_table, episode_rewards, SHIELD_ON, CQL, EPISODES, N_ACTIONS, SAVE_RESULTS, SAVE_Q_TABLE, RS, SHOW_EVERY)

		epsilon = np.interp(episode, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
		lr = np.interp(episode, [0, LEARNING_RATE_DECAY], [LEARNING_RATE_START, LEARNING_RATE_END])

		places_no_walls = no_walls(SIZE, walls)
		if DYNAMIC:
			player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True)
			target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True)
		else:
			if LAYOUT == layout_original_walled_big:
				player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=10, y=15, random_init=False) #big: x=10 y=15
				target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=6, y=1, random_init=False) #x=6, y=1
			else:
				player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=7, y=4, random_init=False) #small: x=5, y=7
				target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=6, y=1, random_init=False)
		# enemy = Agent(places_no_walls, random_init=True)

		if episode % SHOW_EVERY == 0:
			print(f"Episode: {episode}, epsilon: {epsilon}, lr: {lr}, mean reward: {np.mean(episode_rewards[-SHOW_EVERY:])}")

		# steps in episode
		episode_reward = 0
		rs_penalty = 0
		for i in range(100):
			# obs = (player-target, player-enemy)
			obs = (player-target)

			rnd = np.random.random()
			if SHIELD_ON:
				action, rs_penalty, overrided_action = shielded_action(rnd, epsilon, q_table, obs, N_ACTIONS, player, walls, shield,
															RS, RS_PENALTY)
			else:
				action = unshielded_action(rnd, epsilon, q_table, obs, N_ACTIONS)

			reward, done = check_reward(player, target, action, walls, TARGET_REWARD, WALL_PENALTY, MOVE_PENALTY, rs_penalty=rs_penalty)
			
			# next_state_qvalue = player.get_potential_position(action)

			episode_reward += reward

			# perform action
			player.action(action)

			# new_obs = (player-target, player-enemy)
			new_obs = (player-target)
			new_q = calc_new_q(CQL, q_table, obs, new_obs, action, lr, reward, DISCOUNT, player, walls)
			q_table[obs][action] = new_q

			# if true, update (penalize) qvalue illegal state action pair as well (according to shield)
			if rs_penalty < 0:
				q_table[obs][overrided_action] = new_q

			# render visualization
			if SHOW:
				env = gridworld(layout=LAYOUT, size=SIZE)
				env.render(player, target, step=i, reward=reward)

			# if target/wall/enemy is hit reset the game
			if done:
				break
			
		# save reward
		episode_rewards.append(episode_reward)
	
	return episode_rewards
	# plot directly
	# if PLOT:
	# 	plot2("direct plot", episode_rewards, SHOW_EVERY)

if __name__ == "__main__":
	run_experiment()
