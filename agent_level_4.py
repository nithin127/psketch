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


def use_or_movement(world_start, world_end, inventory):
	use = None
	movement = None
	new_inventory = inventory
	start_state = np.where(world_start==1)
	end_state = np.where(world_end==1)	
	start_view = np.where((world_start + 0.5) % 1 == 0)
	end_view = np.where((world_end + 0.5) % 1 == 0)
	if start_view == end_view:
		object_in_front_start = world_start[start_view]
		object_in_front_end = world_end[end_view]
		# Now find out if a usable object was used
		if object_in_front_end == object_in_front_start:
			if object_in_front_start == -0.5:
				use = 0
			elif (object_in_front_start > 1) and (object_in_front_start < 5):
				# Covering the possible change in the inventory
				new_inventory = inventory_change(inventory, int(object_in_front_start + 0.5))
				use = object_in_front_start + 0.5
				movement = get_direction(end_state, end_view)
			elif (object_in_front_start > 5) and (object_in_front_start < 8):
				movement = get_direction(end_state, end_view)
			elif (object_in_front_start > 8) and (object_in_front_start < 10):
				movement = get_direction(end_state, end_view)
				if (object_in_front_start == 8.5) and (inventory["bridge"] == 0):
					use = object_in_front_start + 0.5
				if (object_in_front_start == 9.5) and (inventory["axe"] == 0):
					use = object_in_front_start + 0.5
			elif (object_in_front_start > 10) and (object_in_front_start < 12):
				movement = get_direction(end_state, end_view)
		else:
			# Covering the change in the inventory
			new_inventory = inventory_change(inventory, int(object_in_front_start + 0.5))
			use = object_in_front_start + 0.5
	else:
		movement = get_direction(end_state, end_view)
	return use, movement, new_inventory


def inventory_change(inventory, action):
	if action == 3:
		num_wood = inventory["wood"]
		num_iron_stick = min(inventory["iron"], inventory["stick"])
		num_grass = inventory["grass"]
		inventory["wood"] -= num_wood
		inventory["iron"] -= num_iron_stick
		inventory["stick"] -= num_iron_stick
		inventory["grass"] -= num_grass
		inventory["plank"] += num_wood
		inventory["axe"] += num_iron_stick
		inventory["rope"] += num_grass
	elif action == 4:
		num_wood = inventory["wood"]
		num_grass_plank = min(inventory["grass"], inventory["plank"])
		num_iron_stick = min(inventory["iron"], inventory["stick"])
		inventory["wood"] -= num_wood
		inventory["grass"] -= num_grass_plank
		inventory["plank"] -= num_grass_plank
		inventory["iron"] -= num_iron_stick
		inventory["stick"] -= num_iron_stick
		inventory["stick"] += num_wood
		inventory["bed"] += num_grass_plank
		inventory["shears"] += num_iron_stick
	elif action == 5:
		num_grass = inventory["grass"]
		num_iron_wood = min(inventory["iron"], inventory["stick"])
		num_plank_stick = min(inventory["plank"], inventory["stick"])
		inventory["grass"] -= num_grass
		inventory["iron"] -= num_iron_wood
		inventory["wood"] -= num_iron_wood
		inventory["plank"] -= num_plank_stick
		inventory["stick"] -= num_plank_stick
		inventory["cloth"] += num_grass
		inventory["bridge"] += num_iron_wood
		inventory["ladder"] += num_plank_stick
	elif action == 6:
		inventory["iron"] += 1
	elif action == 7:
		inventory["grass"] += 1
	elif action == 8:
		inventory["wood"] += 1
	elif action == 9:
		if inventory["bridge"] > 0:
			inventory["bridge"] -= 1
	elif action == 10:
		pass
	elif action == 11:
		inventory["gold"] += 1
	elif action == 12:
		inventory["gem"] += 1
	return inventory


string_num_dict = { "w0": 3, "w1": 4, "w2": 5, "iron": 6, "grass": 7, "wood": 8, "water": 9, "stone": 10, "gold": 11, "gem": 12 }
num_string_dict = { 3: "w0", 4: "w1", 5: "w2", 6: "iron", 7: "grass", 8: "wood", 9: "water", 10: "stone", 11: "gold", 12: "gem" }		


def print_event(inventory, prev_inventory):
	Used = []
	Created = []
	for key in inventory.keys():
		change = inventory[key] - prev_inventory[key]
		if change > 0:
			Created.append((change, key))
		elif change < 0:
			Used.append((-change, key))
	if len(Used) + len(Created) > 0:
		print("{} was used to create {}".format(Used, Created))


# --------------------------------------- Agent Function -------------------------------------- #


class Agent_Level_4():
	def __init__(self):
		# Level 3: Agent can see the basic environment usables and workshops distinctly
		# Agent has a sense of direction, and a basic sense of inventory
		self.inventory = { "wood": 0, "iron": 0, "grass": 0, "plank": 0, "stick": 0, "axe": 0, \
				"rope": 0, "bed": 0, "shears": 0, "cloth": 0, "bridge": 0, "ladder": 0, "gem": 0, "gold": 0 }
		self.level_0_discriminators = [ self.basic_discriminator ]
		self.level_1_discriminators = [ self.navigation_discriminator ]
		self.level_2_discriminators = [ self.use_object_discriminator ]
		# We need a way to ensure that use_object_discriminator is lower priority than the others
		

	def observation_function(self, s):
		# Agent Direction, distinct usables, distinct workshops
		# 0 stands for free space
		# 1 stands for agent
		# 2 stands for obstacles
		# 3 stands for w0
		# 4 stands for w1
		# 5 stands for w2
		# 6 stands for iron
		# 7 stands for grass
		# 8 stands for wood
		# 9 stands for water
		# 10 stands for stone
		# 11 stands for gold
		# 12 stands for gem
		final_s = s[:,:,:11].sum(axis=2)*2
		final_s += s[:,:,1]
		final_s += s[:,:,2]*2
		final_s += s[:,:,3]*3
		final_s += s[:,:,6]*4
		final_s += s[:,:,7]*5
		final_s += s[:,:,8]*6
		final_s += s[:,:,4]*7
		final_s += s[:,:,5]*8
		final_s += s[:,:,9]*9
		final_s += s[:,:,10]*10
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
		if not direction == None:
			return (1, direction)
		else:
			possibilities = []
			use, movt, _ = use_or_movement(demo_model[0], demo_model[-1], self.inventory)
			if movt:
				possibilities.append(movt)
			if use:
				possibilities.append((4, use))
			return (1, possibilities)


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
		if len(actions) < len(demo_model) - 1:
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
			use, _, _ = use_or_movement(penultimate_world, end_world, self.inventory)
			if use:
				result, _ = self.navigation_discriminator(demo_model[:-1])
				if result == 1:
					return (1, (use, final_direction))
		else:
			result, _ = self.navigation_discriminator(demo_model)
			if result == 1:
				return (0.5, (None, None))
		return (0, (None, None))


	def skill_predict(self, demo):
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
				seg2, pred2 = self.level_2_discriminators[0](demo[start_id:end_id])
				all_zero = (seg0 + seg1 + seg2 == 0)
				segmentation_array.append([seg0, seg1, seg2])
				prediction_array.append([pred0, pred1, pred2])
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


	def describe_events(self, prev_params):
		params = []
		inventory = self.inventory
		params.append([0, inventory.copy()])
		for param in prev_params:
			if param[1] == 2:
				use_object = param[-1][0][0]
				prev_inventory = inventory.copy()
				inventory = inventory_change(inventory, use_object)
				print_event(inventory, prev_inventory)
				params.append([param[0], inventory.copy()])
		return params


def main():
	# Let's import a map and see
	demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	agent = Agent_Level_4()
	# Convert into agent readable format
	demo_model = [fullstate(s) for s in demos[0][0]]
	basic_prediction = agent.skill_predict(demo_model)
	abstract_predict = agent.describe_events(basic_prediction)


if __name__ == "__main__":
	main()

