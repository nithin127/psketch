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



class EventEncoderGraph():
	def __init__(self):
		self.event_table = np.empty((0, 23)) # 21 is the inventory assumption array, 22 is rank, 23 is index
		self.start_node = EventNode()
		self.start_node.name = "Start Node"
		self.event_nodes = [ self.start_node ]
		self.index = 1
		self.initial_inventory_assumption = np.zeros(21)


	def add_node(self, new_node, condition, transition):
		self.event_nodes.append(new_node)
		if condition.sum() == 0:
			self.start_node.post_links.append(new_node)
			new_node.prev_links.append(self.start_node)
		else:
			# Keep on going. One by one
			inds = self.event_table[:, 22]
			required_objects = set(np.where(condition != 0)[0])
			# Making a shortcut, we know the number of each unique object is one in this case
			# To. Do. Come up with a more general method
			for i, ind in enumerate(inds):
				available_objects = np.where(self.event_table[i][:21] != 0)[0]
				intersection = set(available_objects) & required_objects
				if not intersection:
					continue
				else:
					for obj in intersection:
						# Here is the major assumption that the number of object exchange is 1
						# Check the generality of these statements
						self.event_table[i][obj] += transition[obj]
					required_objects -= intersection
					self.event_nodes[int(ind)].post_links.append(new_node)
					new_node.prev_links.append(self.event_nodes[int(ind)])
			if required_objects:
				for obj in required_objects:
					# Here is the major assumption that the number of object exchange is 1
					self.initial_inventory_assumption[obj] += 1
		index = self.index
		self.index += 1
		node_availability = transition.clip(0)
		value = node_availability.sum() + index*0.01 + sum([prev_node.value for prev_node in new_node.prev_links])
		new_node.value = value
		table_entry = np.expand_dims(np.append(node_availability, np.array([value, index])), axis = 0)
		self.event_table = np.append(self.event_table, table_entry, axis = 0)
		self.reorganise_table()


	def reorganise_table(self):
		# Remove nodes with no availability
		self.event_table = np.delete(self.event_table, np.where(self.event_table[:,:21].sum(axis=1) == 0)[0], 0)
		# Arrange the columns in ascending order of values
		sorted_args = self.event_table[:,-2].argsort()
		self.event_table = self.event_table[sorted_args]


	def print(self):
		print("Haven't written down the functions")


	def key_events(self):
		# Key events are the ones which weren't used to ensure any other event's occurence and
		# ones which can be re-used again
		nodes = []
		for ind in self.event_table[:,-1]:
			nodes.append(self.event_nodes[int(ind)])
		for node in self.event_nodes:
			if len(node.post_links) == 0: nodes.append(node)
		return set(nodes)


	def independent_events(self):
		# Key events are the ones which weren't used to ensure any other event's occurence and
		# ones which can be re-used again
		nodes = []
		for ind in self.event_table[:,-1]:
			nodes.append(self.event_nodes[int(ind)])
		for node in self.event_nodes:
			if len(node.post_links) == 0: nodes.append(node)
		return set(nodes)



class EventNode():
	def __init__(self):
		self.name = "New Node"
		self.prev_links = []
		self.post_links = []
		self.details = None
		self.value = 0



# --------------------------------------- Agent Function -------------------------------------- #


class Agent():
	def __init__(self, environment_handler):
		# Level 3: Agent can see the basic environment usables and workshops distinctly
		# Agent has a sense of direction, and a basic sense of inventory
		self.environment_handler = environment_handler
		# These things can be replaced by neural networks
		self.skills = [ self.navigation, self.use_object ]
		self.discriminators = [ self.navigation_discriminator, self.use_object_discriminator ]
		self.concept_functions = [ ("object_before", self.object_in_front_before), ("object_after", self.object_in_front_after), \
		("new_reachable_objects", self.new_reachable_objects), ("event_location", self.event_location) ]
		# Agent's memory. Permanent to temporary
		self.rule_dict = {}
		self.rule_sequence = []
		self.object_reachability_set_initial = []
		self.object_reachability_set_current = []
		self.events = []
		self.graph = None
		self.current_inventory = np.zeros(21)
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []


	def restart_segmentation(self):
		self.current_state_sequence = [self.current_state_sequence[-1]]
		self.current_segmentation_array = []
		self.current_prediction_array = []


	def restart(self):
		self.current_state_sequence = []
		self.current_segmentation_array = []
		self.current_prediction_array = []
		self.current_inventory = np.zeros(21)
		self.rule_sequence = []
		self.events = []
		self.graph = None


	def next_state(self, state):
		if self.object_reachability_set_initial == []:
			self.object_reachability_set_current = self.update_reachable_object_list(state)
			self.object_reachability_set_initial = self.object_reachability_set_current.copy()
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


	def what_happened(self, make_graph = False):
		# If there are still some unpredicted actions, predict them first
		if self.current_segmentation_array:
			self.predict(True)
		else:
			pass
		self.restart_segmentation()
		# Now let's see what happened in events
		print("------------------------")
		print("   Describing events    ")
		print("------------------------")
		for ie, event in enumerate(self.events):
			if not event["object_before"] in self.rule_dict.keys():
				success = self.environment_handler.train(event, self)
				print("Training agent for event {}".format(event))
				if not success:
					print("Could not find appropriate rules")
					self.rule_sequence.append(None)
					continue
			# Continue execution
			rules, conditions, desc_set = self.rule_dict[event["object_before"]]
			# Check which the conditions are satisfied. And predict the next set of inventories
			# Print the possible events that could've taken place, record the event
			rules_executed = []
			for i, (rule, condition, desc) in enumerate(zip(rules, conditions, desc_set)):
				if ((self.current_inventory - condition >= 0).all()) and \
					((rule[-1] == 0 and event["object_before"] == event["object_after"]) \
						or (rule[-1] == -1 and event["object_after"] == 0)):
					self.current_inventory += rule[:-1]
					print("== Event == {}".format(desc))
					rules_executed.append(i)
			self.rule_sequence += [(event["object_before"], rule) for rule in rules_executed]
		print("------------------------")
		if make_graph:
			self.graph = EventEncoderGraph()
			for i, (event, rule) in enumerate(zip(self.events, self.rule_sequence)):
				transition = self.rule_dict[rule[0]][0][rule[1]]
				condition = self.rule_dict[rule[0]][1][rule[1]]
				name = self.rule_dict[rule[0]][2][rule[1]]
				# Non observable transition
				inventory_transition = transition[:-1]
				# Let's create and fill up the node
				new_node = EventNode()
				new_node.details = rule
				new_node.name = name
				self.graph.add_node(new_node, condition, inventory_transition)
			
		return self.rule_sequence, self.events, self.graph


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


	def update_reachable_object_list(self, state):
		world = self.observation_function(state)
		start = np.where(world == 1)
		# Dijsktra logic
		cost_map = np.inf*np.ones(world.shape)
		dir_map = np.zeros(world.shape)
		cost_map[start[0],start[1]] = 0
		to_visit = []
		to_visit.append(start)
		new_guys = []
		while len(to_visit) > 0:
			curr = to_visit.pop(0)
			for nx, ny, d in find_neighbors(curr, None):
				if world[nx, ny] > 2:
					if (nx[0],ny[0]) not in self.object_reachability_set_current:
						self.object_reachability_set_current.append((nx[0], ny[0]))
						new_guys.append((nx[0], ny[0]))
				cost = cost_map[curr[0],curr[1]] + 1
				if cost < cost_map[nx,ny]:
					if world[nx, ny] == 0 or world[nx, ny] == -0.5:
						to_visit.append((nx, ny))
					dir_map[nx,ny] = d
					cost_map[nx,ny] = cost
		return new_guys


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


	def new_reachable_objects(self, states):
		return self.update_reachable_object_list(states[-1])



def main():
	# Initialise agent and rulebook
	environment_handler = EnvironmentHandler()
	agent = Agent(environment_handler)
	# Pick demos
	#demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
	#demo_model = [ fullstate(s) for s in demos[0][0] ]
	for demo in [pickle.load(open("demos.pk", "rb"))[-1]]:
		num_rules_prev = len(agent.rule_dict)
		demo_model = [ fullstate(s) for s in demo ]
		# Pass the demonstration "online"
		for state in demo_model:
			agent.next_state(state)
		rule_sequence, events, graph = agent.what_happened(make_graph = True)
		# We need to print graph here
		input("{} new rules added\n Key Events in demo: {}\nContinue ?".\
			format(len(agent.rule_dict) - num_rules_prev, [event.name for event in graph.key_events()]))
		input("{} new rules added\n Independent Events in demo: {}\nContinue ?".\
			format(len(agent.rule_dict) - num_rules_prev, [event.name for event in graph.independent_events()]))
		import ipdb; ipdb.set_trace()
		agent.restart()
	#print("Final set of rules: \n\n".format())
	#for i, rule in enumerate(agent.rule_dict):
	#		print("Rule Number:{} || obj:{}\nrules:{}\nconditions:{}\n\n".format(i, rule["object"], rule["rules"], rule["conditions"]))
	import ipdb; ipdb.set_trace()
		


if __name__ == "__main__":
	main()

