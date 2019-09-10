import pickle
import random
import numpy as np

from craft.envs.craft_world import CraftScenario, CraftWorld
from system1 import System1, EnvironmentHandler


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


class Node():
	def __init__(self, rule, key):
		self.rule = rule
		self.key = key
		self.pre_requisites = []
		self.output = []
		self.done = False


# --------------------------------------- Agent Function -------------------------------------- #


class System1Adapted(System1):
	def __init__(self):
		super().__init__()
		self.concept_functions.append(("new_reachable_objects", self.new_reachable_objects))
		self.object_reachability_set_initial = []
		self.object_reachability_set_current = []


	def restart(self):
		self.object_reachability_set_initial = []
		self.object_reachability_set_current = []
		super().restart()


	def next_state(self, state):
		if self.object_reachability_set_initial == []:
			self.object_reachability_set_current = self.new_reachable_objects([state])
			self.object_reachability_set_initial = self.object_reachability_set_current.copy()
		super().next_state(state)


	def new_reachable_objects(self, states):
		state = states[-1]
		world = self.observation_function(state)
		start = np.where(world == 1)
		# Dijsktra logic
		cost_map = np.inf*np.ones(world.shape)
		dir_map = np.zeros(world.shape)
		cost_map[start[0],start[1]] = 0
		to_visit = []
		to_visit.append(start)
		new_objects = []
		while len(to_visit) > 0:
			curr = to_visit.pop(0)
			for nx, ny, d in find_neighbors(curr, None):
				if world[nx, ny] > 2:
					if (nx[0],ny[0]) not in self.object_reachability_set_current:
						self.object_reachability_set_current.append((nx[0], ny[0]))
						new_objects.append((nx[0], ny[0]))
				cost = cost_map[curr[0],curr[1]] + 1
				if cost < cost_map[nx,ny]:
					if world[nx, ny] == 0 or world[nx, ny] == -0.5:
						to_visit.append((nx, ny))
					dir_map[nx,ny] = d
					cost_map[nx,ny] = cost
		return new_objects



class System2():
	def __init__(self):
		self.system1 = None
		# These things can be replaced by neural networks
		self.rule_dict = {}
		self.rule_sequence = []
		self.current_inventory = np.zeros(21)
		self.graph = None


	def restart(self):
		self.system1.restart()
		self.rule_sequence = []
		self.current_inventory = np.zeros(21)


	def what_happened(self, events):
		graph = None
		# Now let's see what happened in events
		print("------------------------")
		print("   Describing events    ")
		print("------------------------")
		for ie, event in enumerate(events):
			if not event["object_before"] in self.rule_dict.keys():
				success = self.system1.environment_handler.train(event, self)
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
		# Let's update the reachability graph (we don't have to)
		self.update_graph()
		return self.rule_sequence


	def update_graph(self):
		initial_inventory = np.zeros(21)
		node_list = []
		condition_table = np.zeros((0,21))
		transition_table = np.zeros((0,22))
		# Initialise stuff
		for key in self.rule_dict.keys():
			for ind in range(len(self.rule_dict[key][0])):
				node_list.append(Node(key, ind))
				transition_table = np.append(transition_table, self.rule_dict[key][0][ind]).reshape(-1, 22)
				condition_table = np.append(condition_table, self.rule_dict[key][1][ind]).reshape(-1, 21)
		# Get pre-requisites
		for i, col in enumerate(condition_table):
			required_objects = np.where(col == 1)[0]
			if len(required_objects) == 0:
				node_list[i].pre_requisites = None
				node_list[i].output = [i]
				node_list[i].done = True
			else:
				for obj in required_objects:
					possible_pretasks = np.where(transition_table[:, obj] == 1)[0]
					if len(possible_pretasks) == 0:
						node_list[i].output = None
						break
					else:
						# Picking a possible pretask at random. Can improve here. Use logic AND, OR
						node_list[i].pre_requisites += [possible_pretasks[0]]
		# Fill up the result list
		todolist = []
		for node_id, node in enumerate(node_list):
			if (not node.done) and (node.output is not None):
				todolist.append((node_id, node))
		# While loop
		for node_id, node in todolist:
			if node.done:
				continue
			for pre_node_id in node.pre_requisites:
				if not node_list[pre_node_id].done:
					todolist.append(node)
					node.output = []
					break
				else:
					node.output += node_list[pre_node_id].output
			node.output += [node_id]
		import ipdb; ipdb.set_trace()
		self.graph = node_list


	def test(self, rule_sequence):
		pass
		


def main():
	# Initialise agent and rulebook
	system1 = System1Adapted()
	system2 = System2()
	# Input playground environment, and link systems
	environment_handler = EnvironmentHandler()
	system1.environment_handler = environment_handler
	system2.system1 = system1
	# Load demos
	for demo in pickle.load(open("demos.pk", "rb")):
		# Let system 1, do the work
		demo_model = [ fullstate(s) for s in demo ]
		for state in demo_model:
			system1.next_state(state)
		segmentation_index, skill_sequence = system1.result()
		# Now system 2, update rules and get result
		num_rules_prev = len(system2.rule_dict)
		rule_sequence, graph = system2.what_happened(skill_sequence)
		# We need to print graph here
		if graph:
			input("{} new rules added\n Key Events in demo: {}\nContinue ?".\
				format(len(system2.rule_dict) - num_rules_prev, [event.name for event in graph.key_events()]))
		else:
			input("{} new rules added\nContinue ?".format(len(system2.rule_dict) - num_rules_prev))
		system2.test(rule_sequence)
		system2.restart()
	#print("Final set of rules: \n\n".format())
	#for i, rule in enumerate(agent.rule_dict):
	#		print("Rule Number:{} || obj:{}\nrules:{}\nconditions:{}\n\n".format(i, rule["object"], rule["rules"], rule["conditions"]))
	import ipdb; ipdb.set_trace()
		


if __name__ == "__main__":
	main()

