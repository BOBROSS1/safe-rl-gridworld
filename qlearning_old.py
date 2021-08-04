import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import time
import random
import importlib
import os
from env import generate_env, layout_original, layout_nowalls


SIZE = 9
LAYOUT = layout_original
SHOW = False
EPISODES = 600000
SHIELD_ON = False
SHIELDED_FUTURE_Q = False
N_ACTIONS = 5 # N_ACTIONS must be 5 or 9 (including standing still)
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
WALL_PENALTY = -10

CHECK_SHIELD_OVERRIDE = False
SHIELD_OVERRIDE_PENALTY = -50

SHOW_EVERY = 2000
SAVE_INTERVAL = 10000

EPSILON_START=1.0
EPSILON_END=0.1 #0.02 # 0.1
EPSILON_DECAY=500000 #1000000

# LEARNING_RATE = 5e-4 #0.1
LEARNING_RATE_START = 0.3
LEARNING_RATE_END = 0.01
LEARNING_RATE_DECAY = 100000

DISCOUNT = 0.95

start_q_table = None # insert qtable filename if available/saved
SAVE_Q_TABLE =  True
SAVE_RESULTS = True


random.seed(34095)
np.random.seed(34095)

if N_ACTIONS == 5:
	original_actions = list(range(4)) + [8]
elif N_ACTIONS==9:
	original_actions = list(range(8)) + [8]
else:
	raise Exception("N_ACTIONS can only be 5 or 9")

# import shield
if SHIELD_ON:
	try:
		mod_name = f"9x9_3_{str(N_ACTIONS - 1)}directions"
		Shield = importlib.import_module(mod_name).Shield
	except ImportError as e:
		print("Could not find shield.")
# else:
#     from no_shield import Shield

full_env = []
for y in range(SIZE):
	for x in range(SIZE):
		full_env.append((y,x))

def get_safe_action(shield, encoded_input):
	corr_action = shield.tick(encoded_input)
	corr_action = int("".join(list(map(str, corr_action[:len(corr_action)-1]))), 2)
	return corr_action

def calc_action_variables(N_ACTIONS):
	'''
	Calculate variables needed to encode N_ACTIONS.

	:param N_ACTIONS: # of actions being used in program
	:returns: # of variables (bits) needed to encode N_ACTIONS
	'''
	return len(bin(N_ACTIONS)[2:])


def save_results(SAVE_RESULTS, SAVE_Q_TABLE):
	if SAVE_RESULTS:
		if SHIELD_ON:
			if SHIELDED_FUTURE_Q:
				with open(f"Results_{EPISODES}_{N_ACTIONS}Actions_Shielded_SHIELDED_FUTQ", "wb") as f:
					pickle.dump(episode_rewards, f)
				print("results saved")
			else:
				with open(f"Results_{EPISODES}_{N_ACTIONS}Actions_Shielded_UNSHIELDED_FUTQ", "wb") as f:
					pickle.dump(episode_rewards, f)
				print("results saved")	
		else:
			with open(f"Results_{EPISODES}_{N_ACTIONS}Actions_Unshielded", "wb") as f:
				pickle.dump(episode_rewards, f)
			print("results saved")

	if SAVE_Q_TABLE:
		if SHIELD_ON:
			with open(f"qtable_{EPISODES}_{N_ACTIONS}Actions_Shielded", "wb") as f:
				pickle.dump(q_table, f)
			print("qtable saved")
		else:
			with open(f"qtable_{EPISODES}_{N_ACTIONS}Actions_Unshielded", "wb") as f:
				pickle.dump(q_table, f)
			print("qtable saved")

class Agent:
	def __init__(self, places_no_walls, x=None, y=None, random_init=True):
		if random_init:
			position = random.choice(places_no_walls)
			self.y = int(position[0])
			self.x = int(position[1])
			places_no_walls.remove(position)
		else:	
			self.x = x
			self.y = y

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)
	
	def state(self):
		return (self.y, self.x)

	def action(self, choice):
		if choice == 0:
			# up
			self.move(y=-1, x=0)
		elif choice == 1:
			# down
			self.move(y=1, x=0)
		elif choice == 2:
			# left
			self.move(y=0, x=-1)
		elif choice == 3:
			# right
			self.move(y=0, x=1)
		elif choice == 4:
			# down left
			self.move(y=1, x=-1)
		elif choice == 5:
			# down right
			self.move(y=1, x=1)
		elif choice == 6:
			# up left
			self.move(y=-1, x=-1)
		elif choice ==7:
			# up right
			self.move(y=-1, x=1)
		elif choice == 8:
			# standing still (has to be last (dfa))
			self.move(y=0, x=0)
	
	def get_potential_position(self, choice):
		if choice == 0:
			# up
			return (self.y - 1, self.x + 0)
		elif choice == 1:
			# down
			return (self.y + 1, self.x + 0)
		elif choice == 2:
			# left
			return (self.y + 0, self.x - 1)
		elif choice == 3:
			# right
			return (self.y + 0, self.x + 1)
		elif choice == 4:
			# down left
			return (self.y + 1, self.x - 1)
		elif choice == 5:
			# down right
			return (self.y + 1, self.x + 1)
		elif choice == 6:
			# up left
			return (self.y - 1, self.x - 1)
		elif choice == 7:
			# up right
			return (self.y - 1, self.x + 1)
		elif choice == 8:
			# standing still (has to be last, (dfa))
			return (self.y, self.x)


	def move(self, x=False, y=False):
		# handle walls (y, x)
		check = (self.y + y, self.x + x)
		if check in walls:
			if SHIELD_ON:
				raise Exception("Shield not working, agent should not make mistakes")
			self.x = self.x
			self.y = self.y
		# handle boundaries env
		elif self.x + x < 0:
			self.x = 0
		elif self.x + x > SIZE-1:
			self.x = SIZE-1
		elif self.y + y < 0:
			self.y = 0
		elif self.y + y > SIZE-1:
			self.y = SIZE-1
		else:
			self.x += x
			self.y += y

# if start_q_table is None:
# 	q_table = {}
# 	for x1 in range(-SIZE + 1, SIZE):
# 		for y1 in range(-SIZE + 1, SIZE):
# 			for x2 in range(-SIZE + 1, SIZE):
# 				for y2 in range(-SIZE + 1, SIZE):
# 					q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5, 0) for i in range(N_ACTIONS)]
if start_q_table is None:
	q_table = {}
	for x1 in range(-SIZE + 1, SIZE):
		for y1 in range(-SIZE + 1, SIZE):
					q_table[(x1,y1)] = [np.random.uniform(-5, 0) for i in range(N_ACTIONS)]

else:
	with open(start_q_table, "rb") as f:
		q_table = pickle.load(f)


episode_rewards = [0]
shield = Shield()
_, walls = generate_env(LAYOUT, SIZE)
for episode in range(EPISODES):
	if episode % SAVE_INTERVAL == 0 and episode > 1:
		save_results(SAVE_RESULTS, SAVE_Q_TABLE)

	epsilon = np.interp(episode, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
	lr = np.interp(episode, [0, LEARNING_RATE_DECAY], [LEARNING_RATE_START, LEARNING_RATE_END])

	places_no_walls = [x for x in full_env if x not in walls]
	player = Agent(places_no_walls)
	food = Agent(places_no_walls, random_init=True)
	# enemy = Agent(places_no_walls)

	if episode % SHOW_EVERY == 0:
		print(f"on # {episode}, epsilon: {epsilon}")
		print(f"on # {episode}, lr: {lr}")  
		print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")

	done = False
	episode_reward = 0
	reward = 0
	# steps in episode
	for i in range(100):
		# obs = (player-food, player-enemy)
		obs = (player-food)

		rnd = np.random.random()
		# actions with shield
		if SHIELD_ON:
			if rnd > epsilon:
				actions = enumerate(q_table[obs])
				actions = sorted(actions, key=lambda x:x[1], reverse=True)
				# print("non random actions: ", actions)
				actions = [x[0] for x in actions]
			else:
				# actions = random_actions_list.copy()
				# random.shuffle(random_actions_list)
				# print(actions)
				actions = random.sample(range(0, N_ACTIONS), 5)
		
			encoded_actions = []
			for a in actions[:3]:
				encoded_actions.append(list(map(int, list(bin(a)[2:].rjust(calc_action_variables(N_ACTIONS), '0')))))
			
			# add sensor simulation (state encoding)
			# original_actions[:-1] --> slice off standing still action (always legal)
			state_enc = []
			for a in original_actions[:-1]:
				player_potential_position = player.get_potential_position(a)
				
				# check for walls, also boundaries env?
				if player_potential_position in walls:
					# print("sensor positive")
					state_enc.append(1)
				else:
					# print("sensor negative")
					state_enc.append(0)
			
			# combine state encoding and actions
			for enc_action in encoded_actions:
				state_enc.extend(enc_action)

			# get safe action from shield
			action = get_safe_action(shield, state_enc)

			# check if shield changed the action
			if CHECK_SHIELD_OVERRIDE:
				if action != actions[0]:
					# filter out standing still actions (4 and 8 are the same ----> MUST be changed (no 8 anymore)
					if actions != 4 and actions[0] != 8: 
						reward = SHIELD_OVERRIDE_PENALTY
			
			# check if player will move into wall (should never happen with shield), not for action 4 (and 8) (standing still)
			if player.get_potential_position(action) in walls:
				if action != 4:
					print("weird, something wrong with shield")
		# actions without shield
		else:
			if rnd > epsilon:
 				action = np.argmax(q_table[obs])
			else:
				# action = random.choice(original_actions)
				action = random.randint(0, N_ACTIONS-1)

		# for detecting wall bumping (has to be before action is performed)
		check_wall_bump = player.get_potential_position(action)

		# move player
		# when N_ACTIONS is 5, action 8 should be used as standing still not 4
		if action == 4 and N_ACTIONS == 5:
			# move player
			player.action(8)
		else:
			player.action(action)


		# move enemy and target?
		# enemy.action(np.random.randint(0, 8))
		# food.action(np.random.randint(0, 8))

		# bump into enemy results in penalty
		# if player.x == enemy.x and player.y == enemy.y:
		# 	reward = -ENEMY_PENALTY
			# done = True
		if player.x == food.x and player.y == food.y:
			reward = FOOD_REWARD
			done = True
		elif check_wall_bump in walls: #and action != N_ACTIONS - 1: # it's not standing still
			# print("TESTSTSSTSTST")
			reward = WALL_PENALTY
			done = True
		# else:
		# 	reward = 0#-MOVE_PENALTY
		
		# new_obs = (player-food, player-enemy)
		new_obs = (player-food)


		if SHIELDED_FUTURE_Q:
			# test with safe q learning algo
			future_steps = list(sorted(enumerate(q_table[new_obs]), key=lambda x:x[1], reverse=True))
			# print("----START-----")
			for step in future_steps:
				# print("values", q_table[new_obs])
				# print("index", fut_reward[0])
				if player.get_potential_position(step[0]) in walls:
					# print('illegal move', fut_reward[0])
					continue
				else:
					max_future_q = q_table[new_obs][step[0]]
					# print("picked", q_table[new_obs][fut_reward[0]])
					break
			# print("----END----")
		else:
			max_future_q = np.max(q_table[new_obs])


		# if N_ACTIONS = 5, and action is 8 this move is on place 4 of the qtable 
		# (otherwise index error because place 8 is not in qtable with 5 actions)
		# if action == 8 and N_ACTIONS == 5:
		# 	print("TESTSTSTSTSTST")
		# 	current_q = q_table[obs][4]
		# else:
		# 	print(action)
		current_q = q_table[obs][action]

		# if reward == FOOD_REWARD:
		# 	new_q = FOOD_REWARD
		# elif reward == -ENEMY_PENALTY:
		# 	new_q = -ENEMY_PENALTY
		# elif reward == -WALL_PENALTY:
		# 	new_q = -WALL_PENALTY
		# elif reward == BIG_SHIELD_PENALTY:
		# 	new_q = -BIG_SHIELD_PENALTY
		# else:
		new_q = (1 - lr) * current_q + lr * (reward + DISCOUNT * max_future_q)

		# if N_ACTIONS = 5, and action is 8 this move must be stored on place 4 of the qtable 
		# (otherwise index error because place 8 is not in qtable with 5 actions)
		# if action == 8 and N_ACTIONS == 5:
		# 	print("TETSTSTSTSTST")
		# 	q_table[obs][4] = new_q
		# else:
		
		q_table[obs][action] = new_q

		if SHOW:
			env, _ = generate_env(LAYOUT, SIZE)

			env[player.y][player.x] = (255, 175, 0, 1) #blue
			env[food.y][food.x] = (0, 255, 0, 1) #green
			# env[enemy.y][enemy.x] = (0, 0, 255, 1) #red

			img = Image.fromarray(env, "RGBA")
			img = img.resize((400, 400))
			cv2.imshow("image", np.array(img))
			cv2.waitKey(1)
			
			if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
				break

		episode_reward += reward

		if done:
			break
		# if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: # or reward == WALL_PENALTY:
			# break
		# print(f"step: {i}")
		# time.sleep(3)
	episode_rewards.append(episode_reward)

	# reset epsilon
	# if epsilon < 0.05:
	# 	epsilon = 0.3
	# else:
		# epsilon *= EPS_DECAY





# plot directly
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("Episode #")
plt.show()