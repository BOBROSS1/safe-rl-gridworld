import matplotlib.pyplot as plt
import pickle
import numpy as np

SHOW_EVERY = 2000

# files = ['Results_60000_5Actions_Shielded_SAFE_FUT', 'Results_60000_5Actions_Shielded_UNSAFE_FUT']
# files = ['Results_10000_9Actions_Shielded_SAFE_FUT', 'Results_10000_9Actions_Shielded_UNSAFE_FUT']
# files = ['Results_600000_5Actions_Shielded_SHIELDED_FUTQ', 'Results_600000_5Actions_Shielded_UNSHIELDED_FUTQ']
# files = ['Results_30001_5Actions_Shielded_SHIELDED_FUTQ_NO_PENALTY', 'Results_30001_5Actions_Shielded_SHIELDED_FUTQ_PENALTY', 
# 		 'Results_30001_5Actions_Shielded_UNSHIELDED_FUTQ_NO_PENALTY', 'Results_30001_5Actions_Shielded_UNSHIELDED_FUTQ_PENALTY']

files = ['Results_10001_5Actions_Shielded_SHIELDED_FUTQ', 'Results_10001_5Actions_Shielded_UNSHIELDED_FUTQ_NO_PENALTY', 
		 'Results_10001_5Actions_Shielded_UNSHIELDED_FUTQ_PENALTY']

		 

for file in files:
	with open(file, "rb") as f:
		results = pickle.load(f)
		moving_avg = np.convolve(results, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
		plt.plot([i for i in range(len(moving_avg))], moving_avg, label=f"{file}")


plt.ylabel(f"Rewards {SHOW_EVERY} steps moving average")
plt.xlabel("Episode #")
plt.title("Fixed starting positions")
plt.legend(loc="center right")
plt.rcParams["figure.figsize"] = (7,10)
plt.show()