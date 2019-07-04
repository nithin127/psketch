import gym
import traceback
import numpy as np
#from generate_demonstrations import generate
from segmentation_inventory import *
from craft.envs import CraftEnv


DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4


def fullstate(state):
    f_state = state.grid[:,:,1:12]
    f_state = np.concatenate((f_state, np.zeros((12,12,1))), axis=2)
    f_state[state.pos[0], state.pos[1], 11] = 1
    if state.dir == 2:   #left
        f_state[state.pos[0] - 1, state.pos[1], 11] = -1
    elif state.dir == 3: #right
        f_state[state.pos[0] + 1, state.pos[1], 11] = -1
    elif state.dir == 1: #up
        f_state[state.pos[0], state.pos[1] + 1, 11] = -1
    elif state.dir == 0: #down
        f_state[state.pos[0], state.pos[1] - 1, 11] = -1
    return f_state


def create_dataset():
	# Create Dataset first

	N = 50
	T_exps_all  = []
	goals_all = []
	env = gym.make("CraftEnv-v0")

	# Gold
	Qe = [1,2,6,7]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [0]
			while not inv_change_expert[-1] == 10:
				state = env.reset(task=((8,10),4)) 				# (Get, Gold), 4 doesn't matter
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]			# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 1:
					inv_change_expert = [0]
			T_exps.append(T_exp)
			goals.append(10)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Gold")

	# Gem
	Qe = [1,5,2,4,8]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			# Lazy as fuck, remove the "try" and fix the thing
			inv_change_expert = [0]
			while not inv_change_expert[-1] == 11:
				state = env.reset(task=((8,11),4)) 				# (Get, Gem), 4 doesn't matter
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]			# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 1:
					inv_change_expert = [0]
			T_exps.append(T_exp)
			goals.append(11)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Gem")


	# Plank
	Qe = [1, 4]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 12:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 12),4)) 							# (Make, Rope)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 2:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(12)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Plank")


	# Stick
	Qe = [1, 5]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 13:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 13),4)) 							# (Make, Rope)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 2:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(13)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Stick")


	# Axe
	Qe = [1,5,2,4]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 14:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 14),4)) 							# (Make, Shear)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 3:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(14)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Axe")


	# Rope
	Qe = [3, 4]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 15:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 15),4)) 							# (Make, Rope)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 2:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(15)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Rope")


	# Bed
	Qe = [1,4,3,5]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 16:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 16),4)) 							# (Make, Shear)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 3:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(16)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Bed")


	# Shear
	Qe = [1,5,2,5]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 17:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 17),4)) 							# (Make, Shear)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 3:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(17)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Shear")


	# Cloth
	Qe = [3, 6]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 18:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 18),4)) 							# (Make, Rope)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 2:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(18)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Cloth")


	# Bridge
	Qe = [2, 1, 6]
	i = 0
	T_exps = []
	goals = []
	while i < N:
		try:
			inv_change_expert = [[0]]
			while not inv_change_expert[-1][-1] == 19:
				if np.random.random() < 0.5:
					state = env.reset(task=((8, 10 + np.random.choice(2)),4)) 		# (Get, Gem/Gold), 4 doesn't matter
				else:
					state = env.reset(task=((1, 19),4)) 							# (Make, Shear)
				for t in T_exps:
					if np.all(t[0].features() == state.features()):
						raise("Sorry yo")
				T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
				inv_change_expert = inventory_changes(T_exp)
				if not len(inv_change_expert[-1]) == 3:
					inv_change_expert = [[0]]
			T_exps.append(T_exp)
			goals.append(19)
			i += 1
		except Exception as e:
			print(e)
			continue

	T_exps_all += T_exps
	goals_all += goals
	print("Done with Bridge")
	

	train_maps = {}
	test_maps = {}

	for key in range(10,20):
		train_maps[key] = [ t[0] for t in T_exps_all[(key-10)*N:(key-10 + 1)*N - 10] ]
		test_maps[key] = [ t[0] for t in T_exps_all[(key-10 + 1)*N - 10:(key-10 + 1)*N] ]

	import pickle
	pickle.dump(train_maps, open("train_maps.pk", "wb"))
	pickle.dump(test_maps, open("test_maps.pk", "wb"))

	Dataset = []

	for i, t in enumerate(T_exps_all):
		for _ in range(5):
			try:
				Q, Seg = predict_sequence(t)
				break
			except:
				continue
		seg_start = 0
		for it, seg_end in enumerate(Seg):
			for s in t[seg_start:seg_end]:
				Dataset.append((s, goals_all[i], Q[it:]))
				seg_start = seg_end
	
	np.random.shuffle(Dataset)
	pickle.dump(Dataset, open("dataset_hrl.pk", "wb"))
	return Dataset


if __name__ == "__main__":
	create_dataset()