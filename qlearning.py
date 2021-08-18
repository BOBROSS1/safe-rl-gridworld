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
import argparse
from env import gridworld, layout_original, layout_original_walled, layout_nowalls, layout_original_walled_big
from helper import *
from agent import Agent


def run_experiment(
	n_experiments = 1,
	agent_type=None,
	SIZE = 11,
	LAYOUT = layout_original_walled,
	SHOW = False,
	EPISODES = 300,
	N_ACTIONS = 5, # N_ACTIONS must be 5 or 9
	# DYNAMIC = True,
	DISCOUNT = 0.95,
	MOVE_PENALTY = -1,
	# ENEMY_PENALTY = -300,
	TARGET_REWARD = 25,
	WALL_PENALTY = -5,

	RS_PENALTY = -1,

	SHOW_EVERY = 50,

	EPSILON_START=0.3,
	EPSILON_END=0.01,

	# LEARNING_RATE = 5e-4 #0.1
	LEARNING_RATE_START = 0.2,
	LEARNING_RATE_END = 0.05,

	start_q_table = None, # insert qtable filename if available/saved
	SAVE_Q_TABLE =  False,
	SAVE_RESULTS = False,
	):
	assert agent_type, "Enter the type of agent you want."

	if agent_type == 'TQLU':
		SHIELD_ON = False
		CQL = False
		RS = False
	elif agent_type ==  'TQLS':
		SHIELD_ON = True
		CQL = False
		RS = False
	elif agent_type ==  'TQLSRS':
		SHIELD_ON = True
		CQL = False
		RS = True
	elif agent_type ==  'CQLS':
		SHIELD_ON = True
		CQL = True
		RS = False
	else:
		AssertionError("Wrong agent name")

	all_episode_rewards = []
	for _ in range(n_experiments):

		SAVE_INTERVAL = EPISODES - 1
		EPSILON_DECAY = EPISODES
		LEARNING_RATE_DECAY = EPISODES - 1

		# checks
		assert N_ACTIONS == 5 or N_ACTIONS == 9, "N_ACTIONS can only be 5 or 9"

		# if trained q table used, only do exploitation and show
		if start_q_table:
			EPSILON_START=0
			EPSILON_END=0
			SHOW=True

		# print settings (for notebook)
		print(f"Agent summary --> Size: {SIZE}, Episodes: {EPISODES}, SHIELD_ON: {SHIELD_ON}, CQL: {CQL}, N_ACTIONS: {N_ACTIONS}, DISCOUNT: {DISCOUNT},\
		MOVE_PENALTY: {MOVE_PENALTY}, TARGET_REWARD: {TARGET_REWARD}, WALL_PENALTY: {WALL_PENALTY}, RS: {RS}, RS_PENALTY: {RS_PENALTY}")
		
		# import shield
		if SHIELD_ON:
			try:
				mod_name = f"shield_{str(N_ACTIONS - 1)}directions"
				Shield = importlib.import_module(mod_name).Shield
				shield = Shield()
			except ImportError as e:
				print("Could not find shield.")

		# qlearning algo
		# q_table = generate_qtable(start_q_table, SIZE, N_ACTIONS)
		# Gen MDP/qtable
		q_table = generate_qtable2(start_q_table, SIZE, N_ACTIONS)


		episode_rewards = [0]
		walls = gridworld(LAYOUT, SIZE).walls

		places_no_walls = no_walls(SIZE, walls)
		player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=5, y=7, random_init=False) #small: x=5, y=7
		target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=6, y=1, random_init=False) #x=6, y=1

		for episode in range(EPISODES):
			# qlearning algo
			# q_table = generate_qtable(start_q_table, SIZE, N_ACTIONS)

			if SAVE_RESULTS:
				if episode % SAVE_INTERVAL == 0 and episode > 1:
					save_results(q_table, episode_rewards, SHIELD_ON, CQL, EPISODES, N_ACTIONS, SAVE_RESULTS, SAVE_Q_TABLE, RS, SHOW_EVERY)

			epsilon = np.interp(episode, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
			lr = np.interp(episode, [0, LEARNING_RATE_DECAY], [LEARNING_RATE_START, LEARNING_RATE_END])

			# places_no_walls = no_walls(SIZE, walls)
			# if DYNAMIC:
			# 	player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True)
				# target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True)
			# else:
				# if LAYOUT == layout_original_walled_big:
					# change starting positions in static big env
					# player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=10, y=15, random_init=False) #big: x=10 y=15
					# target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, x=6, y=1, random_init=False) #x=6, y=1
				# else:
			# player = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True) #small: x=5, y=7
			# target = Agent(places_no_walls, walls, SHIELD_ON, N_ACTIONS, SIZE, random_init=True) #x=6, y=1


			if episode % SHOW_EVERY == 0:
				print(f"Episode: {episode}, epsilon: {epsilon}, lr: {lr}, mean reward: {np.mean(episode_rewards[-SHOW_EVERY:])}")

			# steps in episode
			episode_reward = 0
			rs_penalty = 0
			for i in range(100):
				obs = do_obs(player, target, walls, N_ACTIONS)

				rnd = np.random.random()
				if SHIELD_ON:
					action, rs_penalty, overrided_action = shielded_action(rnd, epsilon, q_table, obs, N_ACTIONS, player, walls, shield,
																RS, RS_PENALTY)
				else:
					action = unshielded_action(rnd, epsilon, q_table, obs, N_ACTIONS)

				reward, done = check_reward(player, target, action, walls, TARGET_REWARD, WALL_PENALTY, MOVE_PENALTY, rs_penalty=rs_penalty)
				episode_reward += reward

				# perform action
				player.action(action)

				new_obs = do_obs(player, target, walls, N_ACTIONS)
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
		all_episode_rewards.append(episode_rewards)

	return all_episode_rewards


def plot_directly(n_experiments, agent_type, N_ACTIONS):
	DF_Agent = pd.DataFrame()
	all_episode_rewards = run_experiment(n_experiments=n_experiments, SAVE_RESULTS=False, agent_type=agent_type, N_ACTIONS=N_ACTIONS)
	for i in range(n_experiments):
		episode_rewards = all_episode_rewards[i]
		tmp_df = pd.DataFrame(episode_rewards)
		DF_Agent = DF_Agent.append(tmp_df)
	DF_Agent.reset_index(inplace=True)
	DF_Agent = DF_Agent[DF_Agent["index"] != 0]

	sns.set_theme(style="darkgrid")
	ax = sns.lineplot(data=DF_Agent, x="index", y=0, err_style="band", legend="brief",label="Agent")
	ax.set(xlabel='Episodes', ylabel="Rewards 100 episodes moving avg", title='')
	plt.show()

def generate_df(all_episode_rewards, plot=False):
	DF_Agent = pd.DataFrame()
	for i in range(len(all_episode_rewards)):
		episode_rewards = all_episode_rewards[i]
		tmp_df = pd.DataFrame(episode_rewards)
		DF_Agent = DF_Agent.append(tmp_df)
	DF_Agent.reset_index(inplace=True)
	DF_Agent = DF_Agent[DF_Agent["index"] != 0]

	if plot:
		sns.set_theme(style="darkgrid")
		ax = sns.lineplot(data=DF_Agent, x="index", y=0, err_style="band", legend="brief",label="Agent")
		ax.set(xlabel='Episodes', ylabel="Rewards 50 episodes moving avg", title='')
		plt.show()

	return DF_Agent


if __name__ == "__main__":
	cli_parser = argparse.ArgumentParser()
	cli_parser.add_argument('--n', action='store', help="If true plot the results directly else return the reward data.",
							required=False, default=1, type=int)
	cli_parser.add_argument('--p', action='store', help="If true plot the results directly else return the reward data.", 
							required=False, default=False, type=bool)
	cli_parser.add_argument('--a', action='store', help="Agent type.", 
							required=False, default='TQLU', type=str)
	cli_parser.add_argument('--n_actions', action='store', help="Agent type.", 
							required=False, default=5, type=int)

	args = vars(cli_parser.parse_args())
	plot_it = args['p']
	n_experiments = args['n']
	agent_type = args['a']
	N_ACTIONS = args['n_actions']

	if plot_it:
		plot_directly(n_experiments=n_experiments, agent_type=agent_type, N_ACTIONS=N_ACTIONS)
	else:
		run_experiment(n_experiments=n_experiments, agent_type=agent_type, N_ACTIONS=N_ACTIONS)