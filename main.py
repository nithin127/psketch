import pickle
import random
import numpy as np


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
			{"object_before": 8, "inventory_before": None, "object_after": 0, "inventory_after": None, "text": "Got wood"},
			{"object_before": 3, "inventory_before": None, "object_after": 3, "inventory_after": None, "text": "Used w0"},
			{"object_before": 7, "inventory_before": None, "object_after": 0, "inventory_after": None, "text": "Got grass"},
			{"object_before": 4, "inventory_before": self.inventory_format.copy(), "object_after": 4, \
				"inventory_after": self.inventory_format.copy(), "text": "Made stick at w1"},
		]
		self.rule_list[-1]["inventory_before"]["wood"] += 1
		self.rule_list[-1]["inventory_after"]["stick"] += 1


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


class EnvironmentHandler():
	def __init__(self):
		# We define the set of environments here
		self.envs = pickle.load(open("envs.pk", "rb"))
		self.inventory_format = { "wood": 0, "iron": 0, "grass": 0, "plank": 0, "stick": 0, "axe": 0, \
				"rope": 0, "bed": 0, "shears": 0, "cloth": 0, "bridge": 0, "ladder": 0, "gem": 0, "gold": 0 }
		self.environment_characteristic = { "wood": 0, "iron": 0, "grass": 0, "w0": 0, "w1": 0, "w2": 0, "water": 0, \
				"stone": 0, "gold": 0, "gem": 0 }
		self.rule_format = {"object_before": 0, "inventory_before": None, "object_after": 0, "inventory_after": None, \
				"text": None}

		
	def train(self, event, prev_events, agent):
		# Replicate the demonstration in different environments
		# 1. Exact replication, in all environments
		demos = []
		prev_event_sequences = []
		for env in self.envs:
			demo = []
			for event_i, store_demo in zip(prev_events + [event], [False]*len(prev_events) + [True]):
				# Prepare environment through previous events
				# Find object location
				world = agent.observation_function(fullstate(env))
				start = np.where(world == 1)
				object_location = np.where(world == event_i["object_before"])
				if len(object_location[0]) > 1:
					object_location = object_location[0][0], object_location[1][0]
				elif len(object_location[0]) == 1:
					actions = agent.skills[event_i["trigger"]](world, start, object_location)
				else:
					continue
				if store_demo: demo.append(env)
				for a in actions:
					_, env = env.step(a)
					if store_demo: demo.append(env)
			# Now generalise !
			demos.append(demo)
			prev_event_sequences.append(prev_events)
		# 2. Randomised replication (change in order of previous events)
		# Skip for now
		# 3. Randomised replication (new events executed in between previous events, dropping off some of the previous events)
		# Skip for now
		self.analyse(event, demos, prev_event_sequences, agent)


	def analyse_state(self, world):
		env_char = self.environment_characteristic.copy()
		for item in range(3,13):
			item_loc = np.where(world == item)
			env_char[num_string_dict[item]] += len(item_loc[0])
		return env_char


	def convert_to_inventory_format(self, inventory):
		formatted_inventory = self.inventory_format.copy()
		items = ["iron", "grass", "wood", "gold", "gem", "plank", "stick", "axe", "rope", "bed", \
			"shears", "cloth", "bridge", "ladder"]
		for i, item in enumerate(items):
            formatted_inventory[item] = inventory[i+7]
		return formatted_inventory


	def analyse(self, event, demos, prev_event_sequences, agent):
		# Items in the environment, at the start
		# Starting inventory  |  The environment is fully observable
		information_storage = {"start_inventory": [], "start_characteristic": [], "end_inventory": [], \
			"end_characteristic": [], "replication": [], "event_details": []}
		for i, (demo, prev_events) in enumerate(zip(demos, prev_event_sequences)):
			no_demo = False
			if demo == []:
				no_demo = True
				demo = [self.envs[i]]
			# Getting relevant information
			information_storage["start_inventory"].append(self.convert_to_inventory_format(demo[0].inventory))
			information_storage["start_characteristic"].append(self.analyse_state(agent.observation_function(fullstate(demo[0]))))
			information_storage["end_inventory"].append(self.convert_to_inventory_format(demo[-1].inventory))
			information_storage["end_characteristic"].append(self.analyse_state(agent.observation_function(fullstate(demo[-1]))))
			if no_demo:
				information_storage["replication"].append(None)
			else:
				agent.current_state_sequence = [fullstate(s) for s in demo]
				concept = agent.describe_actions(event["trigger"])
				agent.reinitialise_current_arrays()
				information_storage["replication"].append(concept == event)
				information_storage["event_details"].append(concept)
		# Now, we distill the information to form a rule
		# Success vs Failure cases
		exceptions = []
		successes = []
		for i, success in enumerate(information_storage["replication"]):
			if success:
				successes.append(i)
			else:
				exceptions.append(i)
		# Now, let's find similarities and differences
		# Let's say this happens easily and we form a rule :p
		# And add this to the agent's rule book
		rule = self.rule_format.copy()
		rule["inventory_before"] = information_storage["start_inventory"][successes[0]]
		rule["inventory_after"] = information_storage["end_inventory"][successes[0]]
		for c_text, _ in agent.concept_functions:
			rule[c_text] = information_storage["event_details"][successes[0]][c_text]
		agent.rulebook.rule_list.append(rule)
		import ipdb; ipdb.set_trace()


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
		self.current_inventory = [ self.inventory_format.copy() ]
		self.rule_sequence = []
		self.events = []


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
				rule_sequence.append(num)
			else:
				print("Unrecoginsed event, back to training")
				# Here we pass the event back to training
				self.environment_handler.train(event, self.events[:ie], self)
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
			if len(actions) >= len(demo_model):
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
	agent.what_happened()


if __name__ == "__main__":
	main()

