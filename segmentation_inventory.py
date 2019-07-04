import os
import gym
import torch
import numpy as np
#from generate_demonstrations import generate
from models.simple_hrl_classic import *
from craft.envs import CraftEnv


### ---------- Helper functions ---------- ###
flatten = lambda l: [item for sublist in l for item in sublist]


def sub_policies(num):
	# These policies are defined in "models.simple_hrl_classic"
	if num == 1:
		return get_wood
	elif num == 2:
		return get_iron
	elif num == 3:
		return get_grass
	elif num == 4:
		return make0
	elif num == 5:
		return make1
	elif num == 6:
		return make2
	elif num == 7:
		return get_gold
	elif num == 8:
		return get_gem


# Note: s.step is using the forward dynamics model
def sample_sequence(Q, start, num_samples=1):
	Ts_list = []
	for _ in range(num_samples):
		Ts = []
		s = start
		Ts.append(s)
		for pi in Q:
			seq = sub_policies(pi)(s)
			if not np.any(seq):
				continue
			for a in seq:
				_, s = s.step(a)
				Ts.append(s)
		Ts_list.append(Ts)
	return Ts_list


def inventory_changes(traj):
	# Here we're only checking the way the inventory changes are progressing; this is very specific to this case
	# and possibly non-transferable	
	inv_change = []
	pr_s = traj[0]
	for s in traj[1:]:
		inv = np.where(s.inventory - pr_s.inventory)[0]
		pr_s = s
		if inv.any(): inv_change.append(inv)
	return inv_change


def get_remaining_sequence(T_exp, s_cr):
	for i, t in enumerate(T_exp):
		if t.features == s_cr.features:
			return T_exp[i:]
	return None


def segment(T_demo, T_scr):
	# Using basic inventory change to identify the segmentation
	# This would return just one value of K and W. But for a start this seems fine
	inv_demo = inventory_changes(T_demo)
	inv_exp = []
	K = []
	W = []
	pr_s = T_scr[0]
	for i, s in enumerate(T_scr[1:]):
		inv = np.where(s.inventory - pr_s.inventory)[0]
		pr_s = s
		if inv.any(): inv_exp.append(inv)
		if len(flatten(inv_demo)) == len(flatten(inv_exp)):
			if np.all(flatten(inv_exp) == flatten(inv_demo)): K.append(s); W.append(i+1); break
		# We can avoid "break"; if we want multiple candidates; but it won't make a difference in this case
	return K,W

### ---------- Main Program ---------- ###


def predict_sequence(T_exp):
		N = [[None, T_exp[0], 0, None]]
		p = 0 										# Count; current node index

		# Forward pass
		while 1:
			if p >= len(N):
				break
			s_cr = N[p][1]							# Current node being considered
			w_cr = N[p][2]							# Cumulative weight so far
			T_scr = get_remaining_sequence(T_exp, s_cr)
			p += 1
			if len(T_scr)<=1:
				continue							# Reached the end of demonstration
			for pi in range(1,9):
				T_new = sample_sequence([pi], s_cr, 5)	# Making sure that we're sampling each policy 5 times, just to make sure #badcoding
				# Cheating
				for it in range(5):
					if (T_new[it][0].inventory == T_new[it][-1].inventory).all():
						continue
					if len(T_new[it]) <= 1:
						continue
					K, W = segment(T_new[it], T_scr)		# Here, we're assuming perfect segmentation; here W returns the index
					for k, w in zip(K,W):
						N.append([s_cr, k, w_cr + w, pi])
					if np.any(K): break
			
		# Backward pass
		Q = []
		Seg = []
		p = np.where([(N[i][1].features() == T_exp[-1].features()).all() for i in range(len(N))])[0][0]
		# To do: Ensure that ".features" doesn't encode
		while N[p][3]:
			Q.append(N[p][3])
			Seg.append(N[p][2])
			p = np.where([(N[i][1].features() == N[p][0].features()).all() for i in range(len(N))])[0][0]

		Q.reverse()
		Seg.reverse()
		return Q, Seg


def main():
	# T = generate()[0] # Just taking 1/100 demonstrations
	# Generate is not working but... let's just use non neural network based demos

	# Assumptions:
	# We don't have to clear obstacles (use primitive objects on the way); there is a clear path, always

	# Expert demo; say Qe = 1,2,6,7
	Qe = [1,2,6,7]
	env = gym.make("CraftEnv-v0")
	state = env.reset(task=((8,10),4)) 			# (Get, Gold), 4 doesn't matter
	inv_change_expert = [0]
	while not inv_change_expert[-1] == 10:
		T_exp = sample_sequence(Qe, state)[0]	# Expert sequence; (not exactly ... constructed using our subpolicies)
		inv_change_expert = inventory_changes(T_exp)

	# Let's say we want to find the sequence Q = 1,2,6,7
	# Assuming we have access to the inventory; to remove asap
	Q, Seg = predict_sequence(T_exp)
	visualise = False
	if Q == Qe:
		input("Great Success, Press any key to continue")
		if visualise:
			while True:
				input("See demo?")
				T_sample = sample_sequence(Q, state)[0]
				for s in T_sample:
					s.render()
					input("Waiting for key press")

	import ipdb; ipdb.set_trace()

if __name__ == "__main__":
	main()


