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


def rule_book(inventory, action):
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


# --------------------------------------- Agent Function -------------------------------------- #


class Agent():
	def __init__(self):
		# Level 3: Agent can see the basic environment usables and workshops distinctly
		# Agent has a sense of direction, and a basic sense of inventory
		self.inventory_format = { "wood": 0, "iron": 0, "grass": 0, "plank": 0, "stick": 0, "axe": 0, \
				"rope": 0, "bed": 0, "shears": 0, "cloth": 0, "bridge": 0, "ladder": 0, "gem": 0, "gold": 0 }
		self.discriminators = [ self.navigation_discriminator, self.use_object_discriminator ]
		self.concept_functions = [ ("object_before", self.object_in_front_before), ("object_after", self.object_in_front_after), \
									("use_condition", self.use_condition) ]
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.current_inventory_probability = [(1, self.inventory_format.copy())]
		self.events = []
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
		if (sum(segs) == 0):
			# reinitialise and store concept triggers
			for segs_i, preds_i in zip(self.current_segmentation_array[::-1], self.current_prediction_array[::-1]):
				if 1 in segs_i:
					ind = segs_i.index(1)
					# Ind is a concept trigger
					# Give this to concept function and reinitialise
					self.describe_actions(ind, preds_i)
					self.reinitialise_current_arrays()
					break
		else:
			self.current_segmentation_array.append(segs)
			self.current_prediction_array.append(preds)	


	def describe_actions(self, ind, pred):
		if ind == 0:
			print("Go to: {}".format(pred))
		elif ind == 1:
			print("Use object {} at {}".format(pred[1][0], pred[1][1]))
		# Instead of appending state_sequence, append the result of the concept function
		concepts = {"trigger": ind}
		for key, c_func in self.concept_functions:
			concepts[key] = c_func(self.current_state_sequence[-2:])
		self.events.append(concepts)


	def what_happened(self):
		# If there are still some unpredicted actions, predict them first
		if self.current_segmentation_array:
			segs_i, preds_i = self.current_segmentation_array[-1], self.current_prediction_array[-1]
			if 1 in segs_i:
				ind = segs_i.index(1)
				self.describe_actions(ind, preds_i)
			else:
				print("None of the skills have been completely executed")
		else:
			pass
		self.reinitialise_current_arrays()
		# Now let's see what happened in events
		for ie, event in enumerate(self.events):
			trigger, concepts = event
			# In our case, there is just one concept function, and we don't make use of the event trigger
			concept_num, concept_dict = concepts[0]
			import ipdb; ipdb.set_trace()
			# Let's see what the rule book says
			if concept_dict["direction"][0] == "same":
				if concept_dict["object"][0] == "same":
					# If it is a workshop like object: then either USE or Movement
					# Change the inventory: split probablities
					# If it is a usable object then only Movement
					pass
				elif concept_dict["object"][1][1] == 0:
						print("Picked up object {}".format(concept_dict["object"][1][0]))
						# Change the inventory
				else:
					print("Created object {} from {}".format(concept_dict["object"][1][1], concept_dict["object"][1][0]))

			else:
				print("Changed direction to: {}".format(concept_dict["direction"][1][1]))
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
		# -0.5 for direction
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
		# Treat everything else as obstacles, no direction: observation level 0
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
		# This function is not really needed, but is for reference. And maybe we'll use it later
		# Treat everything else as obstacles, no direction: observation level 0,
		world = np.clip(world, 0, 2)
		# Dijsktra logic -- starts here, importance on the final direction
		cost_map = np.inf*np.ones(world.shape)
		dir_map = np.zeros(world.shape)
		to_visit = []
		# Deciding pre-goals
		for nx, ny, d in find_neighbors(goal, None):
			if (world[nx, ny] == obstacle_id):
				continue
			cost_map[nx, ny] = 0
			dir_map[nx, ny] = d
			for nxx, nyy, dd in find_neighbors([nx, ny], None):
				if not (cost_map[nxx, nyy] == np.inf) or world[nxx, nyy] == obstacle_id:
					continue
				else:
					if dd == d:
						cost_map[nxx, nyy] = 1
					else:
						cost_map[nxx, nyy] = 2
					to_visit.append((nxx, nyy))
					dir_map[nxx, nyy] = dd
		# Meat of the algorithm
		while len(to_visit) > 0:
			curr = to_visit.pop(0)
			for nx, ny, d in find_neighbors(curr, None):
				if world[nx, ny] == obstacle_id:
					continue
				cost = cost_map[curr[0],curr[1]] + 1
				if cost < cost_map[nx,ny]:
					cost_map[nx,ny] = cost
					to_visit.append((nx, ny))
					dir_map[nx,ny] = d
		seq = []
		curr = start
		while not (curr == goal):
			d = dir_map[curr[0],curr[1]]
			curr = get_prev(curr, d)
			if d == UP:
				seq.append(DOWN)
			elif d == DOWN:
				seq.append(UP)
			elif d == RIGHT:
				seq.append(LEFT)
			elif d == LEFT:
				seq.append(RIGHT)
		if seq[-1] == seq[-2]:
			return seq[:-1] + [4]
		else:
			return seq + [4]


	def use_object_discriminator(self, demo_model):
		if len(demo_model) < 2:
			return (0.5, None)
		# Start state
		start_world = self.observation_function(demo_model[0])
		start_state = np.where(start_world==1)
		# Last state
		ultimate_world = self.observation_function(demo_model[-1])
		ultimate_state = np.where(ultimate_world == 1)
		ultimate_direction = np.where((ultimate_world +0.5) % 1 == 0)
		# Second last state
		penultimate_world = self.observation_function(demo_model[-2])
		penultimate_state = np.where(penultimate_world == 1)
		penultimate_direction = np.where((penultimate_world +0.5) % 1 == 0)
		# Checking conditions and outputting results
		if ultimate_state == penultimate_state:
			actions = self.use_object(start_world, start_state, ultimate_direction)
			if ultimate_direction == penultimate_direction:
				if len(actions) >= len(demo_model) - 1:
					return (1, ultimate_direction)
			else:
				if len(actions) >= len(demo_model) - 1:
					return (0.5, ultimate_direction)
				else:
					return (0, None)
		else:
			actions = self.navigation(start_world, start_state, ultimate_state)
			if len(actions) >= len(demos):
				return (0.5, None)
			else:
				return (0, None)		


	def object_in_front_before(self, states):
		state_obs1 = self.observation_function(states[0])
		direction1 = np.where((state_obs1 + 0.5) % 1 == 0)
		return state_obs1[direction1] + 0.5


	def object_in_front_after(self, states):
		state_obs2 = self.observation_function(states[1])
		direction2 = np.where((state_obs2 + 0.5) % 1 == 0)
		return state_obs2[direction2] + 0.5


	def use_condition(self, states):
		state_obs1 = self.observation_function(states[0])
		position1 = np.where(state_obs1 == 1)
		direction1 = np.where((state_obs1 + 0.5) % 1 == 0)
		state_obs2 = self.observation_function(states[1])
		position2 = np.where(state_obs2 == 1)
		direction2 = np.where((state_obs2 + 0.5) % 1 == 0)
		if (direction1 == direction2) and (position1 == position2):
			return True
		else:
			return 0



def main():
	# Let's import a map and see
	demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	demo_model = [ fullstate(s) for s in demos[0][0] ]
	# Initialise agent
	agent = Agent()
	agent.restart()
	# Pass the demonstration "online"
	for state in demo_model:
		agent.next_state(state)
	agent.what_happened()


if __name__ == "__main__":
	main()

