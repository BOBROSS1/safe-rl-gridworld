import random
import pickle
import numpy as np
import matplotlib as plt

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
                        q_table[(x1,y1)] = [np.random.uniform(-5, 0) for i in range(N_ACTIONS)]
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

def safe_action(rnd, epsilon, obs, q_table, N_ACTIONS, player, walls, shield):
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
    action = apply_shield(shield, state_enc)
    
    return action
    # # check if shield changed the action
    # if CHECK_SHIELD_OVERRIDE:
    #     if action != actions[0]:
    #         # filter out standing still actions (4 and 8 are the same ----> MUST be changed (no 8 anymore)
    #         if actions != 4 and actions[0] != 8: 
    #             reward = SHIELD_OVERRIDE_PENALTY

    # # check if player will move into wall (should never happen with shield), not for action 4 (and 8) (standing still)
    # if player.get_potential_position(action) in walls:
    #     if action != 4:
    #         print("weird, something wrong with shield")

def random_action(rnd, epsilon, q_table, obs, N_ACTIONS):
    if rnd > epsilon:
        action = np.argmax(q_table[obs])
    else:
        action = random.randint(0, N_ACTIONS-1)
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