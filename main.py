import pickle
import random
import numpy as np

from craft.envs.craft_world import CraftScenario, CraftWorld
from craft.envs.cookbook import Cookbook

# -------------------------------------- Helper Functions ------------------------------------- #


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


# ----------------------------------------- Rule Book ----------------------------------------- #


class RuleBook():
	def __init__(self):
		self.rule_structure = {"object_before": None, "inventory_before": None, "object_after": None, "inventory_after": None}
		self.inventory_format = { "wood": 0, "iron": 0, "grass": 0, "plank": 0, "stick": 0, "axe": 0, \
				"rope": 0, "bed": 0, "shears": 0, "cloth": 0, "bridge": 0, "ladder": 0, "gem": 0, "gold": 0 }
		self.rule_list = [
			{"object_before": 8, "inventory_before": self.inventory_init(), "object_after": 0, "inventory_after": self.inventory_init("wood")},
			{"object_before": 3, "inventory_before": self.inventory_init(), "object_after": 3, "inventory_after": self.inventory_init()},
			{"object_before": 7, "inventory_before": self.inventory_init(), "object_after": 0, "inventory_after": self.inventory_init("grass")},
			{"object_before": 4, "inventory_before": self.inventory_init("wood"), "object_after": 4, "inventory_after": self.inventory_init("stick")}
		]


	def inventory_init(self, text = None, num = 1):
		inv = self.inventory_format.copy()
		if text:
			inv[text] += num
		return inv


	def rule_number(self, event, agent_inventory_before):
		# We're assuming no two rules satisfy the same criteria, for now
		# Also, no complications with multiple objects in the inventories, we'll see as we go
		for rule_num, rule in enumerate(self.rule_list):
			if (rule["object_before"] == event["object_before"]) and (rule["object_after"] == event["object_after"]):
				if rule["inventory_before"]:
					if rule["inventory_before"] == agent_inventory_before:
						agent_inventory_before = rule["inventory_after"]
						return rule_num, rule["text"]
					else:
						pass
				else:
					return rule_num, rule["text"]
			else:
				pass
		return None, None


	def add_rule(self):
		return None


string_num_dict = { "free": 0, "w0": 3, "w1": 4, "w2": 5, "iron": 6, "grass": 7, "wood": 8, "water": 9, "stone": 10, "gold": 11, "gem": 12 }
num_string_dict = { 0: "free", 3: "w0", 4: "w1", 5: "w2", 6: "iron", 7: "grass", 8: "wood", 9: "water", 10: "stone", 11: "gold", 12: "gem" }		
inventory_number = {"iron": 7, "grass": 8, "wood": 9, "gold": 10, "gem": 11, "plank": 12, "stick": 13, "axe": 14, \
			"rope": 15, "bed": 16, "shears": 17, "cloth": 18, "bridge": 19, "ladder": 20}
number_inventory = {7: "iron", 8: "grass", 9: "wood", 10: "gold", 11: "gem", 12: "plank", 13: "stick", 14: "axe", \
			15: "rope", 16: "bed", 17: "shears", 18: "cloth", 19: "bridge", 20: "ladder"}




class EnvironmentHandler():
	def __init__(self):
		# We define the set of environments here
		self.envs = pickle.load(open("envs.pk", "rb"))
		self.inventory_format = { "wood": 0, "iron": 0, "grass": 0, "plank": 0, "stick": 0, "axe": 0, \
				"rope": 0, "bed": 0, "shears": 0, "cloth": 0, "bridge": 0, "ladder": 0, "gem": 0, "gold": 0 }
		self.rule_format = {"object_before": 0, "inventory_before": None, "object_after": 0, "inventory_after": None, \
				"text": None}

		
	def train(self, event):
		# Replicate the demonstration in different environments
		cw = CraftWorld()
		grid = np.zeros((WIDTH, HEIGHT, cw.cookbook.n_kinds))
		i_bd = cw.cookbook.index["boundary"]
		grid[0, :, i_bd] = 1
		grid[WIDTH-1:, :, i_bd] = 1
		grid[:, 0, i_bd] = 1
		grid[:, HEIGHT-1:, i_bd] = 1
		grid[5, 5, cw.cookbook.index[num_string_dict[event["object_before"]]]] = 1
		scenario = CraftScenario(grid, (5,6), cw)
		# "dataset"
		state_set = []
		prev_inventory_set = []
		for i in range(7,21):
			inventory = np.zeros(21,)
			inventory[i] = 1
			prev_inventory_set.append(inventory)
			state_set.append(scenario.init(inventory))

		for i in range(7,21):
			for j in range(i+1, 21):
				inventory = np.zeros(21,)
				inventory[i] = 1
				inventory[j] = 1
				prev_inventory_set.append(inventory)
				state_set.append(scenario.init(inventory))
		
		for _ in range(100):
			inventory = np.random.randint(4, size=21)
			prev_inventory_set.append(inventory)
			state_set.append(scenario.init(inventory))

		post_inventory_set = []
		inventory_difference_set = []
		object_in_front_difference_set = []
		for i, ss in enumerate(state_set):
			_, sss = ss.step(4)
			post_inventory_set.append(sss.inventory)
			inventory_difference_set.append(sss.inventory - prev_inventory_set[i])
			object_in_front_difference_set.append(sss.grid[5,5].argmax() - event["object_before"])

		# 0; 1, -1
		# If something else: what object is it dependent on ... what object in inventory equals the thing. Check if it satisfies. 
		# If there are multiple indices that satisfy the condition, check min(x, y, z)

		import ipdb; ipdb.set_trace()
	
		

	def convert_to_inventory_format(self, inventory):
		formatted_inventory = self.inventory_format.copy()
		items = ["iron", "grass", "wood", "gold", "gem", "plank", "stick", "axe", "rope", "bed", \
			"shears", "cloth", "bridge", "ladder"]
		for i, item in enumerate(items):
			formatted_inventory[item] = int(inventory[i+7])
		return formatted_inventory



# --------------------------------------- Agent Function -------------------------------------- #


class Agent():
	def __init__(self, rulebook, environment_handler):
		# Level 3: Agent can see the basic environment usables and workshops distinctly
		# Agent has a sense of direction, and a basic sense of inventory
		self.environment_handler = environment_handler
		self.rulebook = rulebook
		self.inventory_format = { "wood": 0, "iron": 0, "grass": 0, "plank": 0, "stick": 0, "axe": 0, \
				"rope": 0, "bed": 0, "shears": 0, "cloth": 0, "bridge": 0, "ladder": 0, "gem": 0, "gold": 0 }
		# These things can be replaced by neural networks
		self.skills = [ self.navigation, self.use_object ]
		self.discriminators = [ self.navigation_discriminator, self.use_object_discriminator ]
		self.concept_functions = [ ("object_before", self.object_in_front_before), ("object_after", self.object_in_front_after) ]
		# Agent's memory
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.current_inventory = self.inventory_format.copy() 
		self.rule_sequence = []
		self.events = []


	def reinitialise_current_arrays(self):
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.current_inventory = self.inventory_format.copy() 


	def restart(self):
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.current_inventory = self.inventory_format.copy() 
		self.rule_sequence = []
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
					event_i = self.describe_actions(ind)
					self.events.append(event_i)
					if ind == 0:
						print("Go to: {}".format(pred))
					elif ind == 1:
						print("Use object at {}".format(pred))
					self.reinitialise_current_arrays()
					break
		else:
			self.current_segmentation_array.append(segs)
			self.current_prediction_array.append(preds)	


	def describe_actions(self, ind):
		# Instead of appending state_sequence, append the result of the concept function
		concepts = {"trigger": ind}
		for key, c_func in self.concept_functions:
			concepts[key] = c_func(self.current_state_sequence[-2:])
		return concepts


	def what_happened(self):
		# If there are still some unpredicted actions, predict them first
		if self.current_segmentation_array:
			segs_i, preds_i = self.current_segmentation_array[-1], self.current_prediction_array[-1]
			if 1 in segs_i:
				ind = segs_i.index(1)
				event_i = self.describe_actions(ind)
				self.events.append(event_i)
				if ind == 0:
					print("Go to: {}".format(preds_i))
				elif ind == 1:
					print("Use object at {}".format(preds_i))	
			else:
				print("None of the skills have been completely executed")
		else:
			pass
		self.reinitialise_current_arrays()
		# Now let's see what happened in events
		print("------------------------")
		print("   Describing events    ")
		print("------------------------")
		for ie, event in enumerate(self.events):
			num, text = self.rulebook.rule_number(event, self.current_inventory)
			if num:
				print("Event {}. {}".format(ie, text))
				self.rule_sequence.append(num)
			else:
				print("Unrecoginsed event, back to training")
				# Here we pass the event back to training
				success = self.environment_handler.train(event)
				if success:
					print("Successfully updated the rule list, redoing event prediction")
					self.rule_sequence = []
					_ = self.what_happened()
				else:
					print("Couldn't find a rule, skipping event {}".format(ie))
		return self.rule_sequence, self.events


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
			if len(actions) >= len(demo_model):
				return (0.5, None)
			else:
				return (0, None)


	def object_in_front_before(self, states):
		state_obs1 = self.observation_function(states[0])
		direction1 = np.where((state_obs1 + 0.5) % 1 == 0)
		return int((state_obs1[direction1] + 0.5)[0])


	def object_in_front_after(self, states):
		state_obs2 = self.observation_function(states[1])
		direction2 = np.where((state_obs2 + 0.5) % 1 == 0)
		return int((state_obs2[direction2] + 0.5)[0])


	def current_position(self, states):
		state_obs2 = self.observation_function(states[1])
		return np.where(state_obs2 == 1)



def main():
	# Pick a demo
	#demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	#demo_model = [ fullstate(s) for s in demos[0][0] ]
	demo = pickle.load(open("iron_one_demo.pk", "rb"))
	demo_model = [ fullstate(s) for s in demo ]
	# Initialise agent and rulebook
	rulebook = RuleBook()
	environment_handler = EnvironmentHandler()
	agent = Agent(rulebook, environment_handler)
	agent.restart()
	# Pass the demonstration "online"
	for state in demo_model:
		agent.next_state(state)
	rule_sequence, events = agent.what_happened()
	agent.restart()
	import ipdb; ipdb.set_trace()


if __name__ == "__main__":
	main()

