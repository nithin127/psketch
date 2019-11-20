import pickle
import random
import numpy as np

from craft.envs.craft_world import CraftScenario, CraftWorld
from system1 import EnvironmentHandler, find_neighbors
from system2 import System1Adapted, System2

# -------------------------------------- Helper Functions ------------------------------------- #


DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4


WIDTH = 12
HEIGHT = 12


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

class System3():
	def __init__(self, rule_dict):
		self.rule_dict = rule_dict
		random_key = np.random.choice(list(self.rule_dict.keys()))
		# Unwrapping the rule dictionary for dependency graph building operations
		self.rule_dict_transitions_unwrapped = np.zeros((0, len(self.rule_dict[random_key][0][0])))
		self.rule_dict_unwrapped_index = []
		for key in self.rule_dict.keys():
			self.rule_dict_transitions_unwrapped = np.append(self.rule_dict_transitions_unwrapped, self.rule_dict[key][0], axis=0)
			self.rule_dict_unwrapped_index += [(key, i) for i in range(len(self.rule_dict[key][0]))]
		unpassable_transistions_index = np.where(self.rule_dict_transitions_unwrapped[:,-1] == 0)[0]
		self.unpassable_objects = [ self.rule_dict_unwrapped_index[i][0] for i in unpassable_transistions_index ]
		

	def infer_objective(self, rule_sequence, reachability_set_sequence, event_position_sequence):
		# This function outputs a reward vector in terms of events, and a reward vector in terms of the minimal 
		# possible final inventory
		#
		# There are independent events: ones which are not pre-requisites for any other event ( + 2 reward )
		# There are some events which would be independent if we're not taking reachability into account ( + 0.5 reward )
		# There are reusable events: ones which can repeatedly act as a pre-requisite for multiple event ( + 1 reward )
		# Then there is a minimal possible final inventory objective
		#
		# We're assigning parents in a way that number of independent events are max -- independent events, one is not required for another event
		#
		reachability_buffer = []
		initial_reachability_set = reachability_set_sequence[0]
		inventory_condition_buffer = []
		parent_edges_established = {}
		independent_nodes = list(range(len(rule_sequence)))
		for i, (rule, new_reachable_objects, position) in enumerate(zip(rule_sequence, reachability_set_sequence[1:], event_position_sequence)):
			transition =  self.rule_dict[rule[0]][0][rule[1]]
			pre_requisites =  self.rule_dict[rule[0]][1][rule[1]]
			##
			#### Reachibility Condition
			##
			# Check if it was already reachable
			if position in initial_reachability_set:
				pass
			else:
				for node in reachability_buffer:
					if position in node[0]:
						parent_edges_established[node[1]] = (i, "reachability")
						ind = independent_nodes.index(node[1])
						_ = independent_nodes.pop(ind)
			# Check if new objects became reachable
			if new_reachable_objects:
				reachability_buffer.append((new_reachable_objects, i))
			##
			#### Inventory Condition
			##
			# Check if any object are being used in inventory
			inventory_used = np.where(pre_requisites == 1)[0]
			if len(inventory_used) > 0:
				possible_parents_inventory = {}
				for obj_used in inventory_used:
					possible_parents_inventory[obj_used] = []
					for obj, possible_parent in inventory_condition_buffer:
						if obj == obj_used:
							possible_parents_inventory[obj_used].append(possible_parent)
				# Check if the objects being used are being exhausted
				inventory_used_up = np.where(transition[:-1] == -1)[0]
				# Now using all this information:
				#
				#	independent_nodes: nodes that aren't a parent to any other node
				#	possible_parents_inventory: nodes that are be used for gaining reachability
				#	inventory_used_up: objects which are used up
				#
				# Fill up parent_edges_established, without disturbing the independent_nodes list as much as possible
				#
				for obj in possible_parents_inventory.keys():
					parent_assigned = False
					for possible_parent in possible_parents_inventory[obj]:
						if not possible_parent in independent_nodes:
							parent_edges_established[possible_parent] = (i, "inventory")
							parent_assigned = True
							break
					##			 ##
					#### To Do ####
					##			 ##
					# Here we can make copies of the graph, instead, for now, we're choosing random
					if not parent_assigned:
						possible_parent = np.random.choice(possible_parents_inventory[obj])
						parent_edges_established[possible_parent] = (i, "inventory")
						# We can do this!
					ind = independent_nodes.index(possible_parent)
					_ = independent_nodes.pop(ind)
					# Check if the object gets exhausted when used up
					if obj in inventory_used_up:
						ind = possible_parents_inventory[obj].index(possible_parent)
						_ = possible_parents_inventory[obj].pop(ind)
						ind = inventory_condition_buffer.index((obj, possible_parent))
						_ = inventory_condition_buffer.pop(ind)
					
			# Check if any objects are added in the inventory
			inventory_added = np.where(transition[:-1] == 1)[0]
			print(rule, "added", inventory_added, self.rule_dict[rule[0]][2][rule[1]])
			for obj in inventory_added:
				inventory_condition_buffer.append((obj, i))
		# Now we have reusable objects
		# Independent events
		# And the task graph for a demonstration
		reward_vector = np.zeros(21)
		independent_events = []
		for event_num in independent_nodes: independent_events.append(rule_sequence[event_num])
		return [reward_vector, independent_events]


	def get_prerequisite(self, node, graph_nodes, node_count = 0):
		pre_requisite = []
		pre_requisite_node_index = []
		new_node_count = 0
		new_unassigned_nodes = []
		transition_vector = self.rule_dict[node[0]][0][node[1]]
		pre_requisite_vector = self.rule_dict[node[0]][1][node[1]]
		for pre_object in np.where(pre_requisite_vector == 1)[0]:
			possible_pre_events = np.where(self.rule_dict_transitions_unwrapped[:,pre_object] == 1)[0]
			or_event_list = []
			or_event_index_list = []
			for or_event in possible_pre_events:
				event_rule = self.rule_dict_unwrapped_index[or_event]
				or_event_list += [event_rule]
				or_event_index = [i for i, g_node in enumerate(graph_nodes) if g_node == event_rule]
				if len(or_event_index) == 0:
					or_event_index_list += [node_count + new_node_count - 1]
					new_node_count += 1
					new_unassigned_nodes += [event_rule]
				else:
					or_event_index_list += or_event_index
			pre_requisite.append(tuple(or_event_list))
			pre_requisite_node_index.append(tuple(or_event_index_list))
		return pre_requisite, pre_requisite_node_index, new_unassigned_nodes, new_node_count


	def update_core_graph_skeleton(self, objectives, graph_nodes = [], graph_skeleton = {}):
		unassigned_nodes = objectives
		prev_graph_nodes = graph_nodes.copy()
		graph_nodes += objectives
		node_count = len(graph_nodes) + len(objectives)
		while len(unassigned_nodes) > 0:
			node = unassigned_nodes.pop()
			pre_requisite, pre_requisite_node_index, \
					new_unassigned_nodes, new_node_count = self.get_prerequisite(node, graph_nodes, node_count)
			node_count += new_node_count
			unassigned_nodes += new_unassigned_nodes
			graph_nodes += new_unassigned_nodes
			node_index = [i for i, g_node in enumerate(graph_nodes) if g_node == node]
			graph_skeleton[node_index[0]] = pre_requisite_node_index
		return graph_nodes, graph_skeleton, set(graph_nodes) - set(prev_graph_nodes)


	def get_reachability_condition(self, initial_config, goal, free_space_id = 0, obstacle_id = 2):
		usable_objects = []
		init_x, init_y = np.where(initial_config == 1)
		init_x, init_y = init_x[0], init_y[0]
		world = np.clip(initial_config, free_space_id, obstacle_id)
		# Dijsktra logic
		cost_map = np.inf*np.ones(world.shape)
		cost_map[goal[0],goal[1]] = 0
		to_visit = []
		to_visit.append(goal)
		while len(to_visit) > 0:
			curr = to_visit.pop(0)
			for nx, ny, d in find_neighbors(curr, None):
				if world[nx, ny] == obstacle_id:
					obj = int(initial_config[nx, ny])
					# Insane Hard-coding B-)
					# Also considering the same objects multiple times. Hihi
					if not (obj in self.unpassable_objects or obj == obstacle_id):
						usable_objects.append((nx, ny, obj))
					continue
				cost = cost_map[curr[0],curr[1]] + 1
				if cost < cost_map[nx,ny]:
					if world[nx, ny] == free_space_id:
						to_visit.append((nx, ny))
					cost_map[nx,ny] = cost
		return not (cost_map[init_x, init_y] == np.inf), list(set(usable_objects))
		

	def adapt_to_env(self, new_nodes, env_adapted_graph, env_adapted_node_dependency, initial_config):
		new_nodes = [node[0] for node in new_nodes]
		new_keys = []
		for obj in new_nodes:
			# Get positions
			xs, ys = np.where(initial_config == obj)
			env_adapted_graph[obj] = [(x, y) for x, y in zip(xs, ys)]
			# Check reachability
			for x, y in zip(xs, ys):
				reachable, possible_dependencies = self.get_reachability_condition(initial_config, (x, y))
				if not reachable:
					env_adapted_node_dependency[(x,y)] = [ (px, py) for px, py, _ in possible_dependencies]
					for _, _, obj_d in possible_dependencies:
						if (obj_d in new_keys) or (obj_d in new_nodes) or (obj_d in list(env_adapted_graph.keys())):
							pass
						else:
							new_keys.append(obj_d)
			# Add to graph or Add to new keys
			# for _, _, obj in possible_dependencies:
			#	if obj in 
		return new_keys, env_adapted_graph, env_adapted_node_dependency


	def get_subgraph_dependency_graph(self, initial_config, objectives, objective_type="reward"):
		# Let's form the graph skeleton: we're assuming this would not be an AND/OR graph
		if objective_type == "event":
			leftover_keys = objectives
			# Core skeleton
			graph_nodes = []
			graph_skeleton = {}
			# Appended skeleton
			env_adapted_graph = {}
			env_adapted_node_dependency = {}
			# Iteration
			while len(leftover_keys) > 0:
				graph_nodes, graph_skeleton, new_nodes = self.update_core_graph_skeleton(leftover_keys, graph_nodes, graph_skeleton)
				leftover_keys, env_adapted_graph, env_adapted_node_dependency = self.adapt_to_env(new_nodes, env_adapted_graph, env_adapted_node_dependency, initial_config)
		elif objective_type == "reward":
			raise("Haven't implemented")
		else:
			raise("Unrecognised objective type: {}".format(objective_type))
		# Now let's adapt the graph skeleton to the current environment instance and fill in the details
		return self.construct_final_graph(graph_skeleton, graph_nodes, env_adapted_graph, env_adapted_node_dependency) # The costs will be added here


	def construct_final_graph(self, graph_skeleton, graph_nodes, env_adapted_graph, env_adapted_node_dependency):
		import ipdb; ipdb.set_trace()
		final_graph = {}
		for key in graph_skeleton.keys():
			obj = graph_nodes[key][0]
			break
		pass


	def play(self, task_sequence):
		pass



def main():
	# Initialise agent and rulebook
	system1 = System1Adapted()
	system2 = System2()
	system2.rule_dict = pickle.load(open("rule_dict.pk", "rb"))
	system3 = System3(system2.rule_dict)
	# Load demos
	for demo in [pickle.load(open("demos.pk", "rb"))[2]]:
		# Let system 1, do the work
		demo_model = [ fullstate(s) for s in demo ]
		for state in demo_model:
			system1.next_state(state)
		segmentation_index, skill_sequence = system1.result()
		# Now system 2, update rules and get result
		num_rules_prev = len(system2.rule_dict)
		rule_sequence, reachability_set_sequence, event_position_sequence = system2.what_happened(skill_sequence, system1)
		# We need to print graph here
		_, independent_events = system3.infer_objective(rule_sequence, reachability_set_sequence, event_position_sequence)
		graph, nodes = system3.get_subgraph_dependency_graph(system1.observation_function(demo_model[0]), independent_events, "event")
	#print("Final set of rules: \n\n".format())
	#for i, rule in enumerate(agent.rule_dict):
	#		print("Rule Number:{} || obj:{}\nrules:{}\nconditions:{}\n\n".format(i, rule["object"], rule["rules"], rule["conditions"]))
	import ipdb; ipdb.set_trace()
		


if __name__ == "__main__":
	main()
