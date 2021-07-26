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
from env import generate_env, layout

style.use("seaborn-darkgrid")

SIZE = 9
HM_EPISODES = 1000000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
# MAX_STEPS_ALLOWED = 100

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None # insert filename
save_q_table =  False

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

shield_on = False
# import shield
if shield_on:
	try:
		mod_name = "9x9_3"
		Shield = importlib.import_module(mod_name).Shield
	except ImportError as e:
		print("Could not find shield.")
else:
    from no_shield import Shield


# c = {1: (255, 175, 0, 1),
# 	 2: (0, 255, 0, 1),
# 	 3: (0, 0, 255, 1),
# 	 "black": (255,255,255, 1)}

# # (y, x)
# walls = [(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(3,1),(3,2),(6,1),
# 		 (7,1),(6,2),(6,3),(7,3),(5,5),(6,5),(7,6),(5,7),(5,8)]



full_env = []
for y in range(SIZE):
	for x in range(SIZE):
		full_env.append((y,x))

def get_safe_action(shield, encoded_input):
	corr_action = shield.tick(encoded_input)
	corr_action = int("".join(list(map(str, corr_action[:len(corr_action)-1]))), 2)
	return corr_action

class Agent:
	def __init__(self, places_no_walls):
		position = random.choice(places_no_walls)
		self.y = int(position[0])
		self.x = int(position[1])
		places_no_walls.remove(position)

	def __str__(self):
		return f"{self.x}, {self.y}"

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)
	
	def state(self):
		return (self.y, self.x)

	def action(self, choice):
		if choice == 0:
			# right
			self.move(y=0, x=1)
		elif choice == 1:
			# left
			self.move(y=0, x=-1)
		elif choice == 2:
			# up
			self.move(y=1, x=0)
		elif choice == 3:
			# down
			self.move(y=-1, x=0)
		elif choice == 4:
			# up left
			self.move(y=1, x=-1)
		elif choice == 5:
			# up right
			self.move(y=1, x=1)
		elif choice == 6:
			# down left
			self.move(y=-1, x=-1)
		elif choice == 7:
			# down right
			self.move(y=-1, x=1)
		elif choice == 8:
			# choice 8
			# standing still
			self.move(y=0, x=0)
	
	def get_potential_position(self, choice):
		if choice == 0:
			# right
			return (self.y + 0, self.x + 1)
		elif choice == 1:
			# left
			return (self.y + 0, self.x - 1)
		elif choice == 2:
			# up
			return (self.y + 1, self.x + 0)
		elif choice == 3:
			# down
			return (self.y - 1, self.x + 0)
		elif choice == 4:
			# up left
			return (self.y + 1, self.x - 1)
		elif choice == 5:
			# up right
			return (self.y + 1, self.x + 1)
		elif choice == 6:
			# down left
			return (self.y - 1, self.x - 1)
		elif choice == 7:
			# down right
			return (self.y - 1, self.x + 1)
		elif choice == 8:
			# choice 8
			# standing still
			return (self.y, self.x)


	def move(self, x=False, y=False):
		# handle walls (y, x)
		check = (self.y + y, self.x + x)
		if check in walls:
			if shield_on:
				raise Exception("Shield not working, agent should not make mistakes")
			self.x = self.x
			self.y = self.y
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

			# handle env boundaries
			# if self.x < 0:
			# 	self.x = 0
			# elif self.x > SIZE-1:
			# 	self.x = SIZE-1
			
			# if self.y < 0:
			# 	self.y = 0
			# elif self.y > SIZE-1:
			# 	self.y = SIZE-1

if start_q_table is None:
	q_table = {}
	for x1 in range(-SIZE + 1, SIZE):
		for y1 in range(-SIZE + 1, SIZE):
			for x2 in range(-SIZE + 1, SIZE):
				for y2 in range(-SIZE + 1, SIZE):
					q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5, 0) for i in range(9)]
	# print(q_table)

else:
	with open(start_q_table, "rb") as f:
		q_table = pickle.load(f)

episode_rewards = [0]
shield = Shield()
_, walls = generate_env(layout, SIZE)
for episode in range(HM_EPISODES):
	places_no_walls = [x for x in full_env if x not in walls]
	player = Agent(places_no_walls)
	food = Agent(places_no_walls)
	enemy = Agent(places_no_walls)
	show = True
	if episode % SHOW_EVERY == 0:
		print(f"on # {episode}, epsilon: {epsilon}") 
		print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
		# show = True
	# else:
		# show = False

	episode_reward = 0
	# steps
	for i in range(100):
		# if i == MAX_STEPS_ALLOWED:
		# 	reward = -100
		obs = (player-food, player-enemy)
		
		# actions with shield
		if shield_on:
			if np.random.random() > epsilon:
				actions = enumerate(q_table[obs])
				actions = sorted(actions, key=lambda x:x[1], reverse=True)
				actions = [x[0] for x in actions]
			else:
				actions = [np.random.randint(0, 9) for x in range(9)]
			
			encoded_actions = []
			for a in actions[:3]:
				encoded_actions.append(list(map(int, list(bin(a)[2:].rjust(4, '0')))))
			
			# add sensor simulation (state encoding)
			state_enc = []
			# 8 directions
			for a in range(8):
				player_potential_position = player.get_potential_position(a)
				# check for walls, also boundaries env?
				if player_potential_position in walls:
					state_enc.append(1)
				else:
					state_enc.append(0)
			
			# combine state encoding and actions
			for enc_action in encoded_actions:
				state_enc.extend(enc_action)

			# get safe acttion from shield
			action = get_safe_action(shield, state_enc)
		
		# actions without shield
		else:
			if np.random.random() > epsilon:
 				action = np.argmax(q_table[obs])
			else:
				action = np.random.randint(0, 9)
		print("position", player.state())
		print("action:", action)

		# move player
		player.action(action)
		#### maybe
		#enemy.action(np.random.randint(0, 8))
		# food.move()
		####

		
		# bump into enemy results in penalty
		if player.x == enemy.x and player.y == enemy.y:
			reward = -ENEMY_PENALTY
		elif player.x == food.x and player.y == food.y:
			reward = FOOD_REWARD
		else:
			reward = -MOVE_PENALTY
		
		new_obs = (player-food, player-enemy)
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]

		if reward == FOOD_REWARD:
			new_q = FOOD_REWARD
		elif reward == -ENEMY_PENALTY:
			new_q = -ENEMY_PENALTY
		else:
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

		q_table[obs][action] = new_q

		if show:
			env, walls = generate_env(layout, SIZE)

			env[player.y][player.x] = (255, 175, 0, 1) #blue
			env[food.y][food.x] = (0, 255, 0, 1) #green
			env[enemy.y][enemy.x] = (0, 0, 255, 1) #red

			img = Image.fromarray(env, "RGBA")
			img = img.resize((300,300))
			cv2.imshow("", np.array(img))

			if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
				if cv2.waitKey(500) & 0xFF == ord("q"):
					break
			else:
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break

		episode_reward += reward
		if reward == FOOD_REWARD or reward == -ENEMY_PENALTY or reward==-100:
			break
		
		# print(player.state(), enemy.state(), food.state())
		# print(f"step: {i}")
		# time.sleep(.1)

	episode_rewards.append(episode_reward)
	epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("Episode #")
plt.show()

if save_q_table:
	with open(f"qtable {int(time.time())}", "wb") as f:
		pickle.dump(q_table, f)