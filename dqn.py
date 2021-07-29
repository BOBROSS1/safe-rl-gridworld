from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random
import cv2
from PIL import Image
import time
import importlib
from env import generate_env, layout_original, layout_nowalls

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.OBSERVATION_SPACE_VALUES))
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.ACTION_SPACE_SIZE)
        )
    
    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

N_ACTIONS = 5
SHIELD_ON = False
SIZE = 9
LAYOUT = layout_original

if N_ACTIONS == 5:
	original_actions = list(range(4)) + [8]
elif N_ACTIONS==9:
	original_actions = list(range(8)) + [8]
else:
	raise Exception("N_ACTIONS can only be 5 or 9")

start_q_table = None # insert qtable filename if available
save_q_table =  False
save_results = True

# import shield
if SHIELD_ON:
	try:
		mod_name = f"9x9_3_{str(N_ACTIONS - 1)}directions"
		Shield = importlib.import_module(mod_name).Shield
	except ImportError as e:
		print("Could not find shield.")
else:
    from no_shield import Shield

full_env = []
for y in range(SIZE):
	for x in range(SIZE):
		full_env.append((y,x))

_, walls = generate_env(LAYOUT, SIZE)

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

class Agent:
    def __init__(self, SIZE, places_no_walls):
        self.size = SIZE
        position = random.choice(places_no_walls)
        self.y = int(position[0])
        self.x = int(position[1])
        places_no_walls.remove(position)

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

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
            return (self.y-1, self.x+0)
        elif choice == 1:
            # down
            return (self.y+1, self.x+0)
        elif choice == 2:
            # left
            return (self.y+0, self.x-1)
        elif choice == 3:
            # right
            return (self.y+0, self.x+1)
        elif choice == 4:
            # down left
            return (self.y+1, self.x-1)
        elif choice == 5:
            # down right
            return (self.y+1, self.x+1)
        elif choice == 6:
            # up left
            return (self.y-1, self.x-1)
        elif choice == 7:
            # up right
            return (self.y-1, self.x+1)
        elif choice == 8:
            # standing still (has to be last, (dfa))
            return (self.y, self.x)

    def move(self, x=False, y=False):
        # handle walls (y, x)
        check = (self.y + y, self.x + x)
        if check in walls:
            if SHIELD_ON:
                raise Exception("Shield not working, agent should not make mistakes.")
            self.x = self.x
            self.y = self.y
        # handle boundaries env
        elif self.x + x < 0:
            self.x = 0
        elif self.x + x > self.size-1:
            self.x = self.size-1
        elif self.y + y < 0:
            self.y = 0
        elif self.y + y > self.size-1:
            self.y = self.size-1
        else:
            self.x += x
            self.y += y


class Gridworld:
    SIZE = 9
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = 4
    ACTION_SPACE_SIZE = 9

    def reset(self):
        places_no_walls = [x for x in full_env if x not in walls]
        self.player = Agent(SIZE, places_no_walls)
        self.food = Agent(SIZE, places_no_walls)
        self.enemy = Agent(SIZE, places_no_walls)

        self.episode_step = 0

        # if self.RETURN_IMAGES:
        #     observation = np.array(self.generate_image())
        # else:
        observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        
        #enemy.move()
        #food.move()
        
        # if self.RETURN_IMAGES:
        #     new_observation = np.array(self.generate_image())
        # else:
        new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.generate_image()
        img = img.resize((400, 400))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(1)
    
    def generate_image(self):
        env, _ = generate_env(LAYOUT, SIZE)
        env[self.player.y][self.player.x]=(255, 175, 0, 1)
        env[self.food.y][self.food.x]=(0, 255, 0, 1)
        env[self.enemy.y][self.enemy.x]=(0, 0, 255, 1)
        img = Image.fromarray(env, 'RGBA')
        return img
        

env = Gridworld()
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# fill up replay buffer by playing a few rounds randomly
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    # action = env.action_space.sample()
    action = random.randint(0, env.ACTION_SPACE_SIZE)

    new_obs, reward, done = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

# main training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    
    # action
    rnd_sample = random.random()
    if rnd_sample <= epsilon:
        action = random.randint(0, env.ACTION_SPACE_SIZE)
    else:
        action = online_net.act(obs)

    new_obs, reward, done = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += reward

    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    # render game
    env.render()

    # gradient step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses_t = torch.as_tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32)
    actions_t = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(np.asarray([t[4] for t in transitions]), dtype=torch.float32)

    # targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

    # loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=0, index=actions_t)
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # step the online NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update weights in target network from online network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # logger
    if step % 1000 == 0:
        print(f"Step #: {step}, Average reward: {np.mean(rew_buffer)}")
