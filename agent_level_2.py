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


def get_direction(start_state, end_state):
	dx = end_state[0] - start_state[0]
	dy = end_state[1] - start_state[1]
	if not dx == 0:
		if dx == -1:
			return LEFT
		else:
			return RIGHT
	elif not dy == 0:
		if dy == -1:
			return DOWN
		else:
			return UP
	else:
		return None


def use_or_movement(world_start, world_end):
	use = None
	movement = None
	start_state = np.where(world_start==1)
	end_state = np.where(world_end==1)	
	start_view = np.where((world_start+0.5) % 1 == 0)
	end_view = np.where((world_end+0.5) % 1 == 0)
	if start_view == end_view:
		# Now find out if a usable object was used
		object_in_front_start = demo_model[0][start_view]
		object_in_front_end = demo_model[-1][end_view]
		if object_in_front_end == object_in_front_start == 2.5:
			movement = get_direction(end_state, end_view)
		elif object_in_front_end == object_in_front_start == 1.5:
			use = 4
			movement = get_direction(end_state, end_view)
		elif object_in_front_end == object_in_front_start == -0.5:
			use = 4
		elif not object_in_front_end == object_in_front_start:
			use = 4
	else:
		movement = get_direction(end_state, end_view)
	return use, movement


# --------------------------------------- Agent Function -------------------------------------- #


class Agent_Level_2():
	def __init__(self):
		# Level 2: Agent can differentiate between usable and non-usable objects
		# Agent has a sense of direction
		self.level_0_discriminators = [ self.basic_discriminator ]
		self.level_1_discriminators = [ self.navigation_discriminator ]
		self.level_2_discriminators = [ self.use_object_discriminator ]
		

	def observation_function(self, s):
		# Direction, usability, non-usability
		# 0 stands for free space
		# 1 stands for agent
		# 2 stands for obstacles
		# 3 stands for usable objects (Wood, Iron, Grass)
		# -0.5 stands for where the agent is pointing
		final_s = s[:,:,:11].sum(axis=2)*2
		final_s += s[:,:,6:9].sum(axis=2)
		final_s[np.where(s[:,:,11] == 1)] = 1
		final_s[np.where(s[:,:,11] == -1)] += -0.5
		return final_s


	def basic_discriminator(self, demo_model):
		if not len(demo_model) == 2:
			return (0, None)
		demo_model = [ self.observation_function(s) for s in demo_model ]
		start_state = np.where(demo_model[0]==1)
		end_state = np.where(demo_model[-1]==1)
		direction = get_direction(start_state, end_state)
		if direction:
			return (1, direction)
		else:
			return (1, use_or_movement(demo_model[0], demo_model[-1]))


	def navigation(self, world, start, goal, free_space_id = 0, agent_id = 1, obstacle_id = 2):
		# Treat everything else as obstacles: observation level 0
		world = np.clip(world, 0, 2)
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


	def use_object_discriminator(self, demo_model):
		end_world = self.observation_function(demo_model[-1])
		penultimate_world = self.observation_function(demo_model[-2])
		end_state = np.where(end_world == 1)
		penultimate_state = np.where(penultimate_world == 1)
		final_direction = np.where((end_world +0.5) % 1 == 0)
		if end_state == penultimate_state:
			use, _ = use_or_movement(penultimate_world, end_world):
			if use:
				result, _ = self.navigation_discriminator(demo_model[:-1])
				if result == 1:
					return (1, final_direction)
		else:
			result, _ = self.navigation_discriminator(demo_model)
			if result == 1:
				return (0.5, None)
		return (0, None)


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
	agent = Agent_Level_2()
	# Convert into agent readable format
	demo_model = [fullstate(s) for s in demos[0][0]]
	agent.predict(demo_model)


if __name__ == "__main__":
	main()

