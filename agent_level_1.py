import pickle
import random
import numpy as np
from create_dataset import fullstate


# --------------------------------------- Helper Function ------------------------------------- #


DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4


WIDTH = 12
HEIGHT = 12


def neighbors(pos, dirc=None):
	x, y = pos
	neighbors = []
	if x > 0 and (dirc is None or dirc == LEFT):
		neighbors.append((x-1, y, LEFT))
	if y > 0 and (dirc is None or dirc == DOWN):
		neighbors.append((x, y-1, DOWN))
	if x < WIDTH - 1 and (dirc is None or dirc == RIGHT):
		neighbors.append((x+1, y, RIGHT))
	if y < HEIGHT - 1 and (dirc is None or dirc == UP):
		neighbors.append((x, y+1, UP))
	return neighbors


def get_prev(pos, dirc):
	if dirc == 0:
		return (pos[0], pos[1] + 1)
	elif dirc == 1:
		return (pos[0], pos[1] - 1)
	elif dirc == 2:
		return (pos[0] + 1, pos[1])
	elif dirc == 3:
		return (pos[0] - 1, pos[1])


# --------------------------------------- Agent Function -------------------------------------- #


class Agent_Level_1():
	def __init__(self):
		# Level 1: Agent sees everything else as obstacle
		self.level_0_discriminators = [ self.basic_discriminator ]
		self.level_1_discriminators = [ self.navigation_discriminator ]
		

	def observation_function(self, s):
		# 0 stands for free space
		# 1 stands for agent
		# 2 stands for obstacles
		final_s = s[:,:,:11].sum(axis=2)*2
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
			possibilities = [4]
			for x, y, direction in neighbors(end_state):
				if demo_model[-1][x, y] == 2:
					possibilities.append(direction)
			return (1, possibilities)


	def navigation(self, world, start, goal, free_space_id = 0, agent_id = 1, obstacle_id = 2):
		# Check if the request is valid
		if (goal == start) or (world[goal] == obstacle_id):
			return []
		# Dijsktra logic
		cost_map = np.inf*np.ones(world.shape)
		dir_map = np.zeros(world.shape)
		cost_map[start[0],start[1]] = 0
		to_visit = []
		to_visit.append(start)
		while len(to_visit) > 0:
			curr = to_visit.pop(0)
			for nx, ny, d in neighbors(curr, None):
				if world[nx, ny] == obstacle_id:
					continue
				cost = cost_map[curr[0],curr[1]] + 1
				if cost < cost_map[nx,ny]:
					if world[nx, ny] == free_space_id:
						to_visit.append((nx, ny))
					dir_map[nx,ny] = d
					cost_map[nx,ny] = cost
		seq = []
		curr = goal
		d = dir_map[curr[0],curr[1]]
		curr = get_prev(curr, d)
		if not dir_map[curr[0],curr[1]] == d:
			seq.append(d)
		while not curr == start:
			d = dir_map[curr[0],curr[1]]
			curr = get_prev(curr, d)
			seq.append(d)
		seq.reverse()
		return seq	


	def navigation_discriminator(self, demo_model):
		world_level_1 = self.observation_function(demo_model[0])
		start_state = np.where(world_level_1==1)
		end_state = np.where(self.observation_function(demo_model[-1])==1)
		actions = self.navigation(world_level_1, start_state, end_state)
		if len(actions) < len(demo_model) - 2:
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
	agent = Agent_Level_1()
	# Convert into agent readable format
	demo_model = [fullstate(s) for s in demos[0][0]]
	agent.predict(demo_model)


if __name__ == "__main__":
	main()

