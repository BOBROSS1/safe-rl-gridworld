#!/usr/bin/python3
import sys

try:
	args = sys.argv
	filename = args[1]
	n_actions = args[2]
except:
	raise Exception("Provide all arguments: filename, n_actions")

with open(filename, "r") as file:
	lines = file.readlines()
	_, states, n_sensors, n_action_variables, _, _, n_transitions = lines[0].replace("\n","").split(" ")
	max_actions = 2**int(n_action_variables)
	unused_actions = max_actions - int(n_actions) - 1 # -1 for standing still
	n_transitions_used_actions = int(n_transitions) - unused_actions - 1 - 1 # -1 for error trand and 0
	legal_transitions = 0
	illegal_transitions = 0
	
	# print(n_transitions + 3)
	print(n_actions)
	print(unused_actions)

	for i in range(3, n_transitions_used_actions + 3):
		line = lines[i].replace("\n","").split(" ")
		state_trans = line[:2]
		action_enc = line[-int(n_action_variables):]
		action_enc_binary = [1 if int(x)>0 else 0 for x in action_enc]
		action_enc_int = int(''.join(str(x) for x in action_enc_binary), 2)
		state_enc = line[2: 2 + int(n_sensors)] 
		line_in_file = i + 1


		if state_trans == ['1', '1']:
			if int(state_enc[action_enc_int]) > 0:
			# if (str(action_enc_int) in state_enc2):
				print(f"Error line {line_in_file}: {state_trans} {state_enc} {action_enc} --> {action_enc_int}")
				raise Exception(f"Incorrect, state is legal but sensor is postive in the direction of the action (this should be a illegal action).")
			else:
				print(f"Correct: state is legal --> line {line_in_file}: ", state_trans, " ", state_enc, " ", action_enc,
						"-->", action_enc_int)
				legal_transitions += 1
		elif state_trans == ['1', '2']:
			# if str(-action_enc_int) in state_enc2:
			if int(state_enc[action_enc_int]) < 0:
				print(f"Error line {line_in_file}: {state_trans} {state_enc} {action_enc} --> {action_enc_int}")
				raise Exception(f"Incorrect, state is illegal but the sensor in the direction of the action is false (this should be a legal action).")
			else:
				print(f"Correct: state is illegal --> line {line_in_file}", state_trans, " ", state_enc, " ", action_enc,
						"-->", action_enc_int)
				illegal_transitions += 1
		else:
			raise Exception(f"Something wrong with state: {state_trans}")

	print(f'Legal transitions: {legal_transitions}')
	print(f'Illegal transitions: {illegal_transitions}')
	print(f"Total transitions: {n_transitions_used_actions}")
	print('No incorrect state transitions in file')

	assert legal_transitions + illegal_transitions == n_transitions_used_actions,\
			f"legal and illegal transitions should add up to {n_transitions_used_actions}" 
