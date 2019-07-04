# In this script, we use a model to look at a demonstration, 
# see when the agent position doesn't change, and output the appropriate "object_pos"
# when looking at it. 
import pickle
import numpy as np
from create_dataset import fullstate

# This is the "model based policy" for skill classification
env2skill = {2:4, 3:5, 4:6, 7:2, 8:3, 9:1, 5:7, 6:8}


def predict_sequence(demo):
	demo_grid = [fullstate(state) for state in demo]
	Q = []
	seg_index = []
	prev_state = demo_grid[0]
	for i, state in enumerate(demo_grid[1:]):
		if (state[:,:,11] == prev_state[:,:,11]).all():
			seg_index.append(i+1)
			# Code for getting the item
			x, y = np.where(state[:,:,11]==-1)
			try:
				Q.append(env2skill[np.argmax(prev_state[x,y,:]) + 1])
			except:
				print("Skill not found. Appending other stuff")
				# 0: found something new 
				Q.append((0, np.argmax(0, prev_state[x,y,:]) + 1))
		prev_state = state
	return Q, seg_index


def main():
	demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	demo_eg = demos[9][0]
	Q, seg_index = predict_sequence(demo_eg)
	import ipdb; ipdb.set_trace()


if __name__ == "__main__":
	main()
