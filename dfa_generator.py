#!/usr/bin/python3
import itertools

def action_to_binary(action, n_action_variables):
	'''
	Transforms digit into it's binary equivalent

	:param action: integer representing a action
	:param n_action_variables: integer representing # of variables used in dfa
	:returns: list with action as binary like: [0, 1, 1, 0] (action 6)
	'''
	return list(bin(action)[2:].rjust(n_action_variables, '0'))

def create_action_encoding(action_as_binary, n_actions):
	''' 
	Generates action encoding. Example: number 4 is transformed to [-9, 10, -11, -12] (0 1 0 0).
	If bit is 0 the number will have a - sign, else it remains positive. What the number is is 
	dependent on n_actions.

	:param action_as_binary: action encoded as binary by action_to_binary function
	:param n_actions: # of actions being used in program
	:returns: encoded action like [-9, 10, 11, -12] (action 6)
	'''
	return [str(-(idx + len(actions)) if x == '0' else (idx + n_actions))
			for idx, x in enumerate(action_as_binary, 1)]

def calc_action_variables(n_actions):
	'''
	Calculate variables needed to encode n_actions (+1 for standing still).

	:param n_actions: # of actions being used in program
	:returns: # of variables (bits) needed to encode n_actions
	'''
	return len(bin(n_actions + 1)[2:])

# up, down, left, right, up_left, up_right, down_left, down_right
n_actions = 8
n_states = 2 # doesn't work with more than 2 states
n_action_variables = calc_action_variables(n_actions) # 4 if n_actions = 8
with open("walls.dfa", "w") as file:
	actions = list(range(n_actions))
	transitions = []
	
	# create list of lists with all possible sensor detections (dependent on n_actions, the actions determine wich 
	# ways we can move and thus have to check for obstacles). Total number of possible (sensor output) combinations is 2^n_actions
	# format combinations: [[], [combination1], [combination2], [combinatation3], ... ]]
	combinations = [[]]
	for i in actions:
		combs = list(map(list, itertools.combinations(actions, i+1)))
		for j in combs:
			combinations.append(j)
	print(combinations)
	# create the sensor encodings based on all possible sensor outputs, if combination is detected these
	# remain positive in the encoding, the others are negative in the encoding
	for combination in combinations:
		sensor_enc = [str(action+1 if action in combination else -(action+1)) for action in actions]
		
		# create all action encodings for each sensor encoding and append after the sensor encoding
		for action in actions:
			action_as_binary = action_to_binary(action, n_action_variables)
			action_enc = create_action_encoding(action_as_binary, n_actions)
			# action is legal if action is not in combination (if it's in combination, these sensors are detecting a obstacle)
			target_state = 1 if action not in combination else 2
			transitions.append("1 {0} {1} {2}\n".format(target_state, " ".join(sensor_enc), " ".join(action_enc)))

	# add action encoding for standing still (9 now, because the other actions are 1-8) and add to transitions
	action_as_binary = action_to_binary(n_actions, n_action_variables)
	action_enc = create_action_encoding(action_as_binary, n_actions)
	transitions.append("1 1 {0}\n".format(" ".join(action_enc)))

	# add unused transitions (we use 4 action variables, so max 16 (1 1 1 1) actions, we use (0) and 1-9, 10-15 are then unused)
	used_actions = n_actions + 1 # +1 for standing still and +1 because the for loop must start at a unused number
	max_actions = 2**n_action_variables
	for action in range(used_actions, max_actions):
		action_as_binary = action_to_binary(action, n_action_variables)
		action_enc = create_action_encoding(action_as_binary, n_actions)
		transitions.append("1 2 {0}\n".format(" ".join(action_enc)))
		
	# apppend 'bad' state loop under transitions
	transitions.append("2 2\n")

	# first line in file
	# write header & start/end states (format: dfa <n_states> <n_state_variables> <n_action_variables> 1 1 <n_transitions>)
	file.write("dfa {0} {1} {2} 1 1 {3}\n".format(n_states, n_actions, n_action_variables, len(transitions)))

	# write states
	for i in range(1, n_states+1):
		file.write("{0}\n".format(i))

	# write transitions
	file.write("".join(transitions))        

	# write state variable names
	for i in range(1, n_actions+1):
		file.write("{0} sensor_{1}\n".format(i,i))

	# write action variable names (loop starts at unique number of actions (sensors) + 1 (standing still))
	total_used_sensors = n_actions + 1
	for i in range(total_used_sensors, total_used_sensors + n_action_variables):
		file.write("{0} o{1}\n".format(i, total_used_sensors + n_action_variables - i))

	exit()