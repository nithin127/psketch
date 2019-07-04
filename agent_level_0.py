import pickle
import random
import numpy as np
from create_dataset import fullstate


# -------------------------------------- Helper Functions ------------------------------------- #

# ... empty

# --------------------------------------- Agent Function -------------------------------------- #


class Agent_Level_0():
	def __init__(self):
		# Level 0: No obstacles. Agent only sees its position
		self.level_0_discriminators = [ self.basic_discriminator ]
		self.level_1_discriminators = [ self.navigation_discriminator ]


	def observation_function(self, s):
		# 0 stands for free space
		# 1 stands for agent
		final_s = np.zeros(s[:,:,0].shape)
		final_s[np.where(s[:,:,11] == 1)] = 1
		return final_s


	def basic_discriminator(self, demo_model):
		if not len(demo_model) == 2:
			return (0, None)
		demo_model = [ self.observation_function(s) for s in demo_model ]
		start_state = np.where(demo_model[0]==1)
		end_state = np.where(demo_model[-1]==1)
		dx = end_state[0] - start_state[0]
		dy = end_state[1] - start_state[1]
		if not dx == 0:
			if dx == -1:
				return (1, 2)
			else:
				return (1, 3)
		elif not dy == 0:
			if dy == -1:
				return (1, 0)
			else:
				return (1, 1)
		else:
			return (1, 4)


	def navigation_discriminator(self, demo_model):
		demo_model = [ self.observation_function(s) for s in demo_model ]
		start_state = np.where(demo_model[0]==1)
		end_state = np.where(demo_model[-1]==1)
		dx = end_state[0] - start_state[0]
		dy = end_state[1] - start_state[1]
		if abs(dx) + abs(dy) < len(demo_model) - 1:
			return (0, None)
		else:
			return (1, end_state)


	def predict(self, demo):
		params = []
		start_id = 0
		while start_id < len(demo) - 1:
			all_zero = False
			prediction_array = []
			segmentation_array = []
			end_id = start_id + 1
			while not all_zero:
				end_id += 1
				seg0, pred0 = self.level_0_discriminators[0](demo[start_id:end_id])
				seg1, pred1 = self.level_1_discriminators[0](demo[start_id:end_id])
				all_zero = (seg0 + seg1 == 0)
				segmentation_array.append([seg0, seg1])
				prediction_array.append([pred0, pred1])
				if end_id >= len(demo):
					break
			# Getting the position where we had a firm prediction
			ind = end_id - start_id - 1
			while True:
				ind -= 1
				if segmentation_array[ind].count(1) > 0:
					break
			start_id = start_id + ind + 1
			# Get the prediction from the lowest level possible
			level_id = segmentation_array[ind].index(1)
			params.append([start_id, level_id, prediction_array[ind][level_id]])
			print("Segmented at: {}, skill level:{}, parameters: {}".\
				format(start_id, level_id, prediction_array[ind][level_id]))
		return params


def main():
	# Let's import a map and see
	demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	agent = Agent_Level_0()
	# Convert into agent readable format
	demo_model = [fullstate(s) for s in demos[0][0]]
	agent.predict(demo_model)


if __name__ == "__main__":
	main()

