# import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from agent import ShadowAgent

def generate_qtable(start_q_table, SIZE, N_ACTIONS):
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
                q_table[(x1,y1)] = [np.random.uniform(0, 1) for i in range(N_ACTIONS)]
    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)
    return q_table

def calc_new_q(SHIELDED_FUTURE_Q, q_table, obs, new_obs, action, lr, reward, DISCOUNT, player, walls):
    if SHIELDED_FUTURE_Q:
        future_steps = list(sorted(enumerate(q_table[new_obs]), key=lambda x:x[1], reverse=True))
        for step in future_steps:
            if player.get_potential_position(step[0]) in walls:
                continue
            else:
                max_future_q = q_table[new_obs][step[0]]
                break
    else:
        max_future_q = np.max(q_table[new_obs])

    current_q = q_table[obs][action]
    return (1 - lr) * current_q + lr * (reward + DISCOUNT * max_future_q)

def no_walls(SIZE, walls):
    full_env = []
    for y in range(SIZE):
        for x in range(SIZE):
            full_env.append((y,x))
    return [x for x in full_env if x not in walls]

def apply_shield(shield, encoded_input):
	corr_action = shield.tick(encoded_input)
	corr_action = int("".join(list(map(str, corr_action[:len(corr_action)-1]))), 2)
	return corr_action

def calc_action_variables(N_ACTIONS):
	'''
	Calculate variables needed to encode N_ACTIONS.
	'''
	return len(bin(N_ACTIONS)[2:])

def shielded_action(rnd, epsilon, q_table, obs, N_ACTIONS, player, walls, shield, CHECK_SHIELD_OVERRIDE, SHIELD_OVERRIDE_PENALTY):
    if rnd > epsilon:
        actions = enumerate(q_table[obs])
        actions = sorted(actions, key=lambda x:x[1], reverse=True)
        actions = [x[0] for x in actions]
    else:
        actions = np.random.choice(range(0, N_ACTIONS), 5)

    encoded_actions = []
    for a in actions[:5]:
        encoded_actions.append(list(map(int, list(bin(a)[2:].rjust(calc_action_variables(N_ACTIONS), '0')))))

    # add sensor simulation (state encoding)
    # original_actions[:-1] --> slice off standing still action (always legal)
    state_enc = []
    
    # check obstructions in all possible directions (standing still action not needed)
    for a in list(range(N_ACTIONS - 1)):
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
    corr_action = shield.tick(state_enc)
    action = int("".join(list(map(str, corr_action[:len(corr_action)-1]))), 2)
    # action = apply_shield(shield, state_enc)
    
    # check if shield changed the action
    override_penalty = 0
    overrided_action = 0
    if CHECK_SHIELD_OVERRIDE:
        if action != actions[0]:
            override_penalty = SHIELD_OVERRIDE_PENALTY
            overrided_action = actions[0]

    return action, override_penalty, overrided_action

def check_reward(player, target, action, walls, TARGET_REWARD, WALL_PENALTY, MOVE_PENALTY, override_penalty=0):
    done = False
    reward = 0
    next_position = player.get_potential_position(action)

    if override_penalty < 0:
        reward += override_penalty

    # check target
    if next_position[0] == target.y and next_position[1] == target.x:
        reward += TARGET_REWARD
        done = True
    # check wall
    elif next_position in walls:
        reward += WALL_PENALTY
        # done = True
    # else:
    #  	reward = MOVE_PENALTY
    return reward, done

def unshielded_action(rnd, epsilon, q_table, obs, N_ACTIONS):
    if rnd > epsilon:
        action = np.argmax(q_table[obs])
    else:
        action = np.random.randint(0, N_ACTIONS-1)
    return action

def plot(episode_rewards, SHOW_EVERY):
    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("Episode #")
    plt.show()

def save_results(q_table, episode_rewards, SHIELD_ON, SHIELDED_FUTURE_Q, EPISODES, N_ACTIONS, SAVE_RESULTS, SAVE_Q_TABLE):
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

# ---------------------------
# live qlearning functions

def apply_shield(actions, player, walls, shield, N_ACTIONS):
    encoded_actions = []
    for a in actions[:5]:
        encoded_actions.append(list(map(int, list(bin(a)[2:].rjust(calc_action_variables(N_ACTIONS), '0')))))

    # add sensor simulation (state encoding)
    # original_actions[:-1] --> slice off standing still action (always legal)
    state_enc = []
    
    # check obstructions in all possible directions (standing still action not needed)
    for a in list(range(N_ACTIONS - 1)):
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
    corr_action = shield.tick(state_enc)
    safe_action = int("".join(list(map(str, corr_action[:len(corr_action)-1]))), 2)
    return safe_action


def shielded_actions(rnd, epsilon, q_table, t0_obs, N_ACTIONS, player, walls, target, shield):
    if rnd > epsilon:
        actions = enumerate(q_table[t0_obs])
        actions = sorted(actions, key=lambda x:x[1], reverse=True)
        actions = [x[0] for x in actions]
    else:
        actions = np.random.choice(range(0, N_ACTIONS), 5)

    # get safe action for t0 --> t1
    t0_action = apply_shield(actions, player, walls, shield, N_ACTIONS) 

    # get the t1_position and t1_obs at t1
    t1_position = player.get_potential_position(t0_action)
    shadow_agent = ShadowAgent(y=t1_position[0], x=t1_position[1], N_ACTIONS=N_ACTIONS)
    t1_obs = shadow_agent - target

    # get the safe action for t1 --> t2
    future_actions = [x[0] for x in sorted(enumerate(q_table[t1_obs]), key=lambda x:x[1], reverse=True)]
    t1_action = apply_shield(future_actions, shadow_agent, walls, shield, N_ACTIONS)

    return t0_action, t1_action, t1_obs
