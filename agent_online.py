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


def find_neighbors(pos, dirc=None):
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
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.events = []
		self.inventory = { "wood": 0, "iron": 0, "grass": 0, "plank": 0, "stick": 0, "axe": 0, \
				"rope": 0, "bed": 0, "shears": 0, "cloth": 0, "bridge": 0, "ladder": 0, "gem": 0, "gold": 0 }
		self.discriminators = [ self.navigation_discriminator, self.use_object_discriminator ]
		self.concept_function = [ self.object_in_front, self.inventory ]
		# We need a way to ensure that use_object_discriminator is lower priority than the others


	def reinitialise_current_arrays(self):
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []


	def restart(self):
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.events = []


	def next_state(self, state):
		self.current_state_sequence.append(state)
		self.predict()
		

	def predict(self):
		preds = []
		segs = []
		for i, disc in enumerate(self.discriminators):
			seg, pred = disc(self.current_state_sequence)
			preds.append(pred)			
			segs.append(seg)			
		print("Segs: {}, Preds:{}".format(segs, preds))
		if (sum(segs) == 0):
			# reinitialise and store concept triggers
			for segs_i, preds_i in zip(self.current_segmentation_array[::-1], self.current_prediction_array[::-1]):
				if 1 in segs_i:
					ind = segs_i.index(1)
					# Ind is a concept trigger
					# Give this to concept function and reinitialise
					self.describe_actions(ind, preds_i)
					self.reinitialise_current_arrays()
		else:
			self.current_segmentation_array.append(segs)
			self.current_prediction_array.append(preds)	


	def describe_actions(self, ind, pred):
		if ind == 0:
			print("Go to: {}".format(pred))
		elif ind == 1:
			print("Use object at: {}".format(pred[1]))
		# Instead of appending state_sequence, append the result of the concept function
		self.events.append((ind, self.current_state_sequence[-2:]))


	def what_happened(self):
		# If there are still some unpredicted actions, predict them first
		segs_i, preds_i = self.current_segmentation_array[-1], self.current_prediction_array[-1]
		if 1 in segs_i:
			ind = segs_i.index(1)
			self.describe_actions(ind, preds_i)
			self.reinitialise_current_arrays()
		else:
			print("None of the skills have been completely executed")
		import ipdb; ipdb.set_trace()
		self.restart()
		return None


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
			for nx, ny, d in find_neighbors(curr, None):
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
		if len(demo_model) < 2:
			return (0.5, None)
		world_level_1 = self.observation_function(demo_model[0])
		start_state = np.where(world_level_1==1)
		end_state = np.where(self.observation_function(demo_model[-1])==1)
		actions = self.navigation(world_level_1, start_state, end_state)
		if len(actions) < len(demo_model) - 1:
			return (0, None)
		else:
			return (1, end_state)


	def use_object(self, world, start, goal, obstacle_id = 2):
		# Change the way we want to see the world (this is one of limiting factors of this function)
		world = np.clip(world, 0, 2)
		# First pick a position next to the object
		neighbors = find_neighbors(goal)
		available_neighbors = []
		for nx, ny, _ in neighbors:
			if not world[nx, ny] == obstacle_id:
				available_neighbors.append((nx,ny))
		if len(available_neighbors) == 0:
			print("No neighbours available")
			return []
		else:
			navigation_goal = available_neighbors[0]
		# Go to the position
		seq = self.navigation(world, start, navigation_goal)
		# Check direction, append use and return the sequence
		required_direction = get_direction(navigation_goal, goal)
		if required_direction == seq[-1]:
			return seq + [4]
		else:
			return seq + [required_direction] + [4]


	def use_object_discriminator(self, demo_model):
		# This function is not perfect, we need to incorporate the change in direction
		# that is possible between the penultimate and the pen-penultimate object
		if len(demo_model) < 2:
			return (0.5, None)
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


	def object_in_front(self, state):
		state_obs = self.observation_function(state)
		final_direction = np.where((state_obs + 0.5) % 1 == 0)
		return state_obs[final_direction] + 0.5


def main():
	# Let's import a map and see
	demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	demo_model = [ fullstate(s) for s in demos[0][0] ]
	# Initialise agent
	agent = Agent_Level_4()
	agent.restart()
	# Pass the demonstration "online"
	for state in demo_model:
		agent.next_state(state)
	agent.what_happened()


if __name__ == "__main__":
	main()

