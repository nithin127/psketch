import pickle
import random
import numpy as np

from craft.envs.craft_world import CraftScenario, CraftWorld

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


string_num_dict = { "free": 0, "workshop0": 3, "workshop1": 4, "workshop2": 5, "iron": 6, "grass": 7, "wood": 8, "water": 9, "stone": 10, "gold": 11, "gem": 12 }
num_string_dict = { 0: "free", 3: "workshop0", 4: "workshop1", 5: "workshop2", 6: "iron", 7: "grass", 8: "wood", 9: "water", 10: "stone", 11: "gold", 12: "gem" }		
inventory_number = {"iron": 7, "grass": 8, "wood": 9, "gold": 10, "gem": 11, "plank": 12, "stick": 13, "axe": 14, \
			"rope": 15, "bed": 16, "shears": 17, "cloth": 18, "bridge": 19, "ladder": 20}
number_inventory = {7: "iron", 8: "grass", 9: "wood", 10: "gold", 11: "gem", 12: "plank", 13: "stick", 14: "axe", \
			15: "rope", 16: "bed", 17: "shears", 18: "cloth", 19: "bridge", 20: "ladder"}


class EnvironmentHandler():
	def __init__(self):
		self.cw = CraftWorld()

	def get_env(self):
		goal = np.random.randint(14) + 7
		scenario = self.cw.sample_scenario_with_goal(goal)
		# Assuming no initial inventory
		return scenario.init()
		
	def train(self, event, agent):
		# Replicate the demonstration in different environments
		grid = np.zeros((WIDTH, HEIGHT, self.cw.cookbook.n_kinds))
		i_bd = self.cw.cookbook.index["boundary"]
		grid[0, :, i_bd] = 1
		grid[WIDTH-1:, :, i_bd] = 1
		grid[:, 0, i_bd] = 1
		grid[:, HEIGHT-1:, i_bd] = 1
		grid[5, 5, self.cw.cookbook.index[num_string_dict[event["object_before"]]]] = 1
		scenario = CraftScenario(grid, (5,6), self.cw)
		# "dataset"
		state_set = []
		for i in range(7,21):
			inventory = np.zeros(21, dtype=int)
			inventory[i] = 1
			state_set.append(scenario.init(inventory))

		for i in range(7,21):
			for j in range(i+1, 21):
				inventory = np.zeros(21, dtype=int)
				inventory[i] = 1
				inventory[j] = 1
				state_set.append(scenario.init(inventory))

		for _ in range(100):
			inventory = np.random.randint(4, size=21)
			state_set.append(scenario.init(inventory))

		prev_inventory_set = np.empty((0, 21))
		difference_set = np.empty((0, 22))

		for i, ss in enumerate(state_set):
			_, sss = ss.step(4)
			# object_in_front_difference should only be -1 or 0, or it is disaster
			object_in_front_difference = np.clip(sss.grid[5,5].argmax() - ss.grid[5,5].argmax(), -1, 1)
			transition = np.expand_dims(np.append(sss.inventory - ss.inventory, object_in_front_difference), axis = 0)
			prev_inventory_set  = np.append(prev_inventory_set, np.expand_dims(ss.inventory, axis = 0), axis = 0)
			difference_set = np.append(difference_set, transition, axis = 0)

		unique_transitions = np.unique(difference_set, axis = 0)
		# We want: the simplest core set of transitions, and the minimum conditions required for them to occur
		# First we arrange them in the order of simplicity
		# and find the minimum conditions

		costs = np.zeros(len(unique_transitions))
		for i, tr in enumerate(unique_transitions):
			costs[i] += abs(tr[7:12]).sum()
			costs[i] += 2*abs(tr[12:]).sum()
		sorted_indices = costs.argsort()
		# Now we get the core transitions 
		core_transitions = np.empty((unique_transitions[0].shape[0], 0), dtype = int)
		pre_requisite_set = np.empty((0, 21), dtype = int)
		desc_set = []
		for ind in sorted_indices:
			matrix = np.append(core_transitions, np.expand_dims(unique_transitions[ind].copy(), axis=1), axis = 1)
			if np.linalg.matrix_rank(matrix) == matrix.shape[1]:
				core_transitions = matrix.copy()
				# Also find the pre-requisite condition
				tr_indices = np.where((unique_transitions[ind] == difference_set).all(axis=1))[0]
				prev_inventory_subset = np.empty((0, 21))
				for tr_ind in tr_indices:
					prev_inventory_subset  = np.append(prev_inventory_subset, np.expand_dims(prev_inventory_set[tr_ind], axis=0), axis = 0)
				pre_requisite = np.min(prev_inventory_subset, axis = 0)
				pre_requisite_set = np.append(pre_requisite_set, np.expand_dims(pre_requisite, axis = 0), axis = 0)
				# Coming up with the description of the event
				objs_gathered = np.where(unique_transitions[ind] == 1)[0]
				objs_used_up = np.where(unique_transitions[ind][:-1] == -1)[0]
				text_gathered = ""
				text_used_up = ""
				for obj in objs_gathered:
					text_gathered += number_inventory[obj] + ", "
				for obj in objs_used_up:
					text_used_up += number_inventory[obj] + ", "
				if len(text_gathered) > 0: 
					text_gathered = text_gathered[:-2] 
				else: 
					text_gathered = None
				if len(text_used_up) > 0: 
					text_used_up = text_used_up[:-2]
				else: 
					text_used_up = None
				# Now for the description
				if unique_transitions[ind][-1] == -1:
					if text_gathered:
						desc_set.append("Got: {}. Used up: {}".format(text_gathered, text_used_up))
					else:
						desc_set.append("Removed {} from the environment. Used up: {}".\
							format(num_string_dict[event["object_before"]], text_used_up))
				else:
					desc_set.append("Used {} to make {} at {}".\
						format(text_used_up, text_gathered, num_string_dict[event["object_before"]]))

		try:
			agent.rule_dict[event["object_before"]] = (core_transitions.T, pre_requisite_set, desc_set)
			return True
		except:
			return False


# --------------------------------------- Agent Function -------------------------------------- #


class System1():
	def __init__(self):
		self.environment_handler = EnvironmentHandler()
		# These things can be replaced by neural networks
		self.skills = [ self.navigation, self.use_object ]
		self.discriminators = [ self.navigation_discriminator, self.use_object_discriminator ]
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.concept_functions = [ ("object_before", self.object_in_front_before), ("object_after", self.object_in_front_after), \
		("event_location", self.event_location) ]
		self.events = []
		self.segmentation_indices = []
		


	def restart_segmentation(self):
		self.current_state_sequence = [self.current_state_sequence[-1]]
		self.current_segmentation_array = []
		self.current_prediction_array = []


	def restart(self):
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.events = []
		

	def next_state(self, state):
		self.current_state_sequence.append(state)
		self.segment()


	def segment(self):
		preds = []
		segs = []
		for i, disc in enumerate(self.discriminators):
			seg, pred = disc(self.current_state_sequence)
			preds.append(pred)
			segs.append(seg)
		if (sum(segs) == 0):
			self.segmentation_indices.append(len(self.current_state_sequence))
			self.predict()
		else:
			self.current_segmentation_array.append(segs)
			self.current_prediction_array.append(preds)	


	def predict(self, final_sequence = False):
		# reinitialise and store concept triggers
		for segs_i, preds_i in zip(self.current_segmentation_array[::-1], self.current_prediction_array[::-1]):
			if 1 in segs_i:
				ind = segs_i.index(1)
				# Ind is a concept trigger
				# Give this to concept function and reinitialise
				event_i = self.describe_actions(ind, final_sequence)
				self.events.append(event_i)
				if ind == 0:
					print("Go to: {}".format(preds_i[ind]))
				elif ind == 1:
					print("Use object at {}".format(preds_i[ind]))
				self.restart_segmentation()
				break


	def describe_actions(self, ind, final_sequence = False):
		# Instead of appending state_sequence, append the result of the concept function
		concepts = {"trigger": ind}
		for key, c_func in self.concept_functions:
			if final_sequence:
				concepts[key] = c_func(self.current_state_sequence[-2:])
			else:
				concepts[key] = c_func(self.current_state_sequence[-3:-1])
		return concepts


	def result(self):
		# If there are still some unpredicted actions, predict them first
		if self.current_segmentation_array:
			self.predict(True)
		else:
			pass
		self.restart_segmentation()
		# Now let's see what happened in events
		return self.segmentation_indices, self.events


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
		start_state = np.where(world_level_1 == 1)
		end_state = np.where(self.observation_function(demo_model[-1]) == 1)
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
		if len(seq) > 1 and seq[-1] == seq[-2]:
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


	def event_location(self, states):
		state_obs2 = self.observation_function(states[1])
		direction2 = np.where((state_obs2 + 0.5) % 1 == 0)
		return (direction2[0][0], direction2[1][0])


	def test(self, rule_sequence):
		envs = []
		demos = []
		for _ in range(100):
			env = self.environment_handler.get_env()
			envs.append(env)
			demo = [env]
			for rule in rule_sequence:
				# Get object location where skill can be executed
				world = self.observation_function(fullstate(env))
				x, y = np.where( world == rule[0])	
				try:
					goal = x[0], y[0]
				except:
					demo = "Goal {} not found".format(rule[[0]])
				x, y = np.where( world == 1)
				start = x[0], y[0]
				# Get sequence of actions in the environment
				try:
					action_seq = self.use_object(world, start, goal, obstacle_id = 2)
					# Execute in the environment
					for a in action_seq:
						_, env = env.step(a)
						demo.append(env)
				except:
					demo = "Couldn't execute action sequence"
			demos.append(demo)
			# We can do more to check the result here. But maybe later. Let's get the definitions right
		self.test_results = {"envs": envs, "demos": demos}




def main():
	# Initialise system
	system = System1()
	# Give playground environment and demo
	environment_handler = EnvironmentHandler()
	system.environment_handler = environment_handler
	# Find skills
	## 	In our work, we assume an exhaustive set of skills are already given
	# Train skill discriminators
	##	In our work, we assume these are already trained properly
	# Load and pass demonstration "online"
	demo = pickle.load(open("demos.pk", "rb"))[-1]
	demo_model = [ fullstate(s) for s in demo ]
	for state in demo_model:
		system.next_state(state)
	# Let's get the output of the system
	segmentation_index, skill_sequence = system.result()
	import ipdb; ipdb.set_trace()
		


if __name__ == "__main__":
	main()
