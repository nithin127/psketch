import copy
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
    f_state = np.concatenate((f_state, np.zeros((f_state.shape[0], f_state.shape[1], 1))), axis=2)
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
	def __init__(self, inventory = None, state_space = None, graph = None):
		self.inventory = inventory.copy()
		self.state_space = state_space.copy()
		self.graph = copy.deepcopy(graph)
		self.skills_so_far = []
		self.cost_so_far = 0
		self.reward_so_far = 0


# --------------------------------------- Agent Function -------------------------------------- #		

class System3():
	def __init__(self, rule_dict):
		#
		# Defining rule dict. Unwrapping the rule dictionary for dependency graph building operations
		#
		self.rule_dict = rule_dict
		random_key = np.random.choice(list(self.rule_dict.keys()))
		self.rule_dict_transitions_unwrapped = np.zeros((0, len(self.rule_dict[random_key][0][0])))
		self.rule_dict_unwrapped_index = []
		for key in self.rule_dict.keys():
			self.rule_dict_transitions_unwrapped = np.append(self.rule_dict_transitions_unwrapped, self.rule_dict[key][0], axis=0)
			self.rule_dict_unwrapped_index += [(key, i) for i in range(len(self.rule_dict[key][0]))]
		unpassable_transistions_index = np.where(self.rule_dict_transitions_unwrapped[:,-1] == 0)[0]
		self.unpassable_objects = [ self.rule_dict_unwrapped_index[i][0] for i in unpassable_transistions_index ]
		#
		# Definition for inferring demo objective
		#
		self.minimum_initial_inventory = np.zeros(len(self.rule_dict[random_key][1][0]))
		self.independent_events = []
		self.reusable_events = []
		self.required_inventory = np.zeros(len(self.rule_dict[random_key][1][0]))
		self.bonus_inventory = np.zeros(len(self.rule_dict[random_key][1][0]))
		self.graph_guide = None
		

	def reset_demo_objectives(self):
		
		random_key = np.random.choice(list(self.rule_dict.keys()))
		self.minimum_initial_inventory = np.zeros(len(self.rule_dict[random_key][1][0]))
		self.independent_events = []
		self.reusable_events = []
		self.required_inventory = np.zeros(len(self.rule_dict[random_key][1][0]))
		self.bonus_inventory = np.zeros(len(self.rule_dict[random_key][1][0]))
		self.graph_guide = None


	def infer_objective(self, rule_sequence, reachability_set_sequence, event_position_sequence):
		
		# This function outputs a reward vector in terms of events, and a reward vector in terms of the minimal 
		# possible final inventory
		#
		# There are independent events: ones which are not pre-requisites for any other event ( + 2 reward )
		# There are some events which would be independent if we're not taking reachability into account ( + 0.5 reward )
		# There are reusable events: ones which can repeatedly act as a pre-requisite for multiple event ( + 1 reward )
		# Then there is a minimal possible final inventory objective
		#
		# (To do)
		# We're assigning parents in a way that number of independent events are max -- independent events, one is not required for another event
		#
		# End result:	- List of -- Independent events, resuable events
		# 				- Minimum initial inventory
		# 				- Required inventory condition, bonus inventory condition
		
		reachability_buffer = []
		initial_reachability_set = reachability_set_sequence[0]
		inventory_condition_buffer = []
		parent_edges_established = {}
		independent_nodes = list(range(len(rule_sequence)))
		self.reset_demo_objectives()
		
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
					# A node cannot become reachable twice, it can have only one parent
					# Every node must have been reachable before it was executed
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
				# Assign possible parents
				possible_parents_inventory = {}
				for obj_used in inventory_used:
					possible_parents_inventory[obj_used] = []
					for obj, possible_parent in inventory_condition_buffer:
						if obj == obj_used:
							possible_parents_inventory[obj_used].append(possible_parent)
					# If there are no parents possible, the object must have been present from before
					if len(possible_parents_inventory[obj_used]) == 0:
						self.minimum_initial_inventory[obj_used] += 1
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
				for obj_used in possible_parents_inventory.keys():
					parent_assigned = False
					for possible_parent in possible_parents_inventory[obj_used]:
						if not possible_parent in independent_nodes:
							parent_edges_established[possible_parent] = (i, "inventory")
							parent_assigned = True
							break
					##			 ##
					#### To Do ####
					##			 ##
					# Here we're randomly assigning parents, we can keep an account of all the nodes yet to be assigned parents
					# And make assignments at the end such that number of independent nodes are "maximum"
					# Skipping for now. 
					if not parent_assigned:
						possible_parent = np.random.choice(possible_parents_inventory[obj_used])
						parent_edges_established[possible_parent] = (i, "inventory")
						# We can do this!
					ind = independent_nodes.index(possible_parent)
					_ = independent_nodes.pop(ind)
					# Check if the object gets exhausted when used up
					if obj_used in inventory_used_up:
						ind = possible_parents_inventory[obj_used].index(possible_parent)
						_ = possible_parents_inventory[obj_used].pop(ind)
						ind = inventory_condition_buffer.index((obj_used, possible_parent))
						_ = inventory_condition_buffer.pop(ind)

			# Check if any objects are added in the inventory
			inventory_added = np.where(transition[:-1] == 1)[0]
			print(rule, "added", inventory_added, self.rule_dict[rule[0]][2][rule[1]])
			for obj in inventory_added:
				# Adding the parent once for each time an object has been added
				for _ in range(int(transition[obj])):
					inventory_condition_buffer.append((obj, i))
		# Now we have reusable objects
		# Independent events
		# And the task graph for a demonstration
		reward_vector = np.zeros(21)
		self.independent_events = []
		self.reusable_events = []
		# Get the independent events and the required inventory
		for event_num in independent_nodes:
			event = rule_sequence[event_num]
			self.independent_events.append(event)
			for obj in np.where(self.rule_dict[event[0]][0][event[1]] == 1)[0]:
				self.required_inventory[obj] += 1
				# Remove the objects related to independent events as they are already considered
				ind = inventory_condition_buffer.index((obj, event_num))
				_ = inventory_condition_buffer.pop(ind)
		# Assign all the remaining inventory objectives as bonus
		for obj, event_num in inventory_condition_buffer: 
			self.reusable_events.append(rule_sequence[event_num])
			self.bonus_inventory[obj] += 1
		return {"independent_events": self.independent_events, "reusable_events": self.reusable_events, \
				"required_inventory": self.required_inventory, "bonus_inventory": self.bonus_inventory}


	def get_prerequisite(self, node, graph_nodes, node_count = 0):
		# Not sure how general is the AND/OR graph
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
					or_event_index_list += [node_count + new_node_count]
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
		node_count = len(graph_nodes)
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
		costs = []
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
				cost = cost_map[curr[0],curr[1]] + 1
				if cost < cost_map[nx,ny]:
					cost_map[nx,ny] = cost
					if world[nx, ny] == obstacle_id:
						obj = int(initial_config[nx, ny])
						# Insane Hard-coding B-)
						# Also considering the same objects multiple times. Hihi
						if not obj == obstacle_id:
							usable_objects.append(((nx, ny), obj, curr))
							costs.append(cost)
						continue
				
					if world[nx, ny] == free_space_id:
						to_visit.append((nx, ny))
					
		return not (cost_map[init_x, init_y] == np.inf), usable_objects, costs


	def adapt_to_env(self, nodes, env_adapted_graph, env_adapted_node_dependency, initial_config):
		new_nodes = []
		for obj, _ in nodes:
			# Get positions
			xs, ys = np.where(initial_config == obj)
			env_adapted_graph[obj] = [(x, y) for x, y in zip(xs, ys)]
			# Check reachability
			# And update graph "appropriately"... lol
			for x, y in zip(xs, ys):
				reachable, possible_dependencies, _ = self.get_reachability_condition(initial_config, (x, y))
				if not reachable:
					env_adapted_node_dependency[(x,y)] = []
					for obj_pos , obj, _ in possible_dependencies:
						if not obj in self.unpassable_objects:
							rules = np.where(self.rule_dict[obj][0][:,-1] == -1)[0]
							for rule in rules:
								env_adapted_node_dependency[(x,y)].append((obj_pos ,(obj, rule)))
								if ((obj, rule) in new_nodes) or ((obj, rule) in nodes) or (obj in list(env_adapted_graph.keys())):
									pass
								else:
									new_nodes.append((obj, rule))
		return new_nodes, env_adapted_graph, env_adapted_node_dependency


	def get_dependency_graph_guide(self, initial_config):
		# Let's form the graph skeleton: we're assuming this would not be an AND/OR graph
		leftover_keys = self.independent_events + self.reusable_events
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
		# Now let's adapt the graph skeleton to the current environment instance and fill in the details
		self.graph_guide = {"skeleton": graph_skeleton, "nodes": graph_nodes, "node_pos": env_adapted_graph, \
							"node_pos_dependencies": env_adapted_node_dependency}
		return self.graph_guide


	def graph_cost(self, node, alpha=0.3, gamma=0.9):
		# There are some repeated calculations each time we try to figure out the cost
		# but for now, we make do with this
		# 
		# alpha: independent to reusable/bonus event reward ratio
		# beta: cost/reward ratio that we'll care for
		# gamma: same as RL. discount factor (used to propagate rewards through the graph)
		#
		inventory = node.inventory
		graph = node.graph
		inventory_reward = (self.required_inventory - inventory).clip(0) + alpha*(self.bonus_inventory - inventory).clip(0)
		nodes = []
		rewards = []
		event__rewards = []
		non_reachable_env_node__rewards = []
		for obj in np.where(inventory_reward > 0)[0]:
			rule_indices =  np.where(self.rule_dict_transitions_unwrapped[:, obj] >  0)[0]
			rules = [self.rule_dict_unwrapped_index[rule_index] for rule_index in rule_indices]
			for rule in rules: 
				event__rewards.append((rule, inventory_reward[obj]))
		while (len(event__rewards) > 0) or (len(non_reachable_env_node__rewards) > 0):
			# This is a nice visualisation, comment this out later
			# input("{}\n{}\n{}\n\n".format(event__rewards, non_reachable_env_node__rewards, nodes))
			# Going through core graph skeleton
			if len(event__rewards) > 0:
				event, reward = event__rewards.pop()
				# Now we propagate reward and get nodes
				# If inventory condition satisfied
				if (inventory - self.rule_dict[event[0]][1][event[1]] >= 0).all():
					# Consider the environment specific position nodes
					env_nodes = graph["node_pos"][event[0]]
					for env_node in env_nodes:
						if env_node in graph["node_pos_dependencies"].keys():
							non_reachable_env_node__rewards.append((env_node, reward*gamma))
						else:
							nodes.append(env_node)
							rewards.append(reward)
				else:
					# add children to list :D
					event_id = graph["nodes"].index(event)
					for or_event_list in graph["skeleton"][event_id]:
						for or_event in or_event_list:
							event = graph["nodes"][or_event]
							# We're not adding nodes that are already satisfied, 
							# and we're subtracting rewards from nodes that are partially satisfied
							for obj in np.where(self.rule_dict[event[0]][0][event[1]] > 0)[0]:
								already_satisfied = inventory[obj]
							new_reward = reward*gamma - already_satisfied
							if new_reward > 0:
								event__rewards.append((event, new_reward))
							else:
								pass
			else:
				pass
			# Going through environment specific nodes
			if len(non_reachable_env_node__rewards) > 0:
				non_reachable_env_node, reward = non_reachable_env_node__rewards.pop()
				possible_events = graph["node_pos_dependencies"][non_reachable_env_node]
				event_found = False
				event__rewards_appendage = []
				for dep_obj_pos, possible_event in possible_events:
					event__rewards_appendage.append((possible_event, reward))
					if (inventory - self.rule_dict[possible_event[0]][1][possible_event[1]] >= 0).all():
						nodes.append(dep_obj_pos)	
						rewards.append(reward)
						event_found = True
						break
					else:
						pass
				if not event_found:
					event__rewards += event__rewards_appendage
			else:
				pass
		# Remove duplicates here, keep the ones with higher costs
		final_nodes = []
		final_rewards = []
		for node, reward in zip(nodes, rewards):
			if not node in final_nodes:
				final_nodes.append(node)
				final_rewards.append(reward)
			else:
				ind = final_nodes.index(node)
				final_rewards[ind] = max(final_rewards[ind], reward)
		return final_nodes, final_rewards


	def get_next_options(self, node, use_graph_guide, beta=0.5):
		# 
		# alpha: independent to reusable/bonus event reward ratio
		# beta: cost/reward ratio that we'll care for #### Not used ####
		# gamma: same as RL. discount factor (used to propagate rewards through the graph)
		#
		config = node.state_space
		init_x, init_y = np.where(config == 1)
		init_x, init_y = init_x[0], init_y[0]
		_, options, costs = self.get_reachability_condition(config, (init_x, init_y))
		if not use_graph_guide:
			combined_list = list(zip(options, costs))
			np.random.shuffle(combined_list)
			return zip(*combined_list)
		else:
			nodes, rewards = self.graph_cost(node)
			sorted_ind = sorted(range(len(rewards)), key=lambda k: rewards[k], reverse=True)
			final_options = []
			final_costs = []
			## Sorting doesn't matter here; but still
			for ind in sorted_ind:
				option, cost = [(options[i], costs[i]) for i in range(len(options)) if nodes[ind] == options[i][0]][0]
				final_options.append(option)
				final_costs.append(cost)
			return final_options, final_costs


	def approximate_transition_step(self, node, option, cost, use_graph_guide):
		# Use rule dict here
		change_inventory = False
		change_structure = False
		obj = option[1]
		obj_pos = option[0]
		agent_pos = option[2]
		new_state = node.state_space.copy()
		new_inventory = node.inventory.copy()
		new_graph = copy.deepcopy(node.graph)
		num_rules = len(self.rule_dict[obj][0])

		for i in range(num_rules):
			transition = self.rule_dict[obj][0][i]
			condition = self.rule_dict[obj][1][i]

			if (new_inventory - condition >= 0).all():
				new_inventory += transition[:-1]
				change_inventory = True
			else:
				pass

			if transition[-1] == -1:
				new_state[obj_pos[0], [obj_pos[1]]] = 0
				ind = new_graph["node_pos"][obj].index(obj_pos)
				_ = new_graph["node_pos"][obj].pop(ind)
				_ = new_graph["node_pos_dependencies"].pop(obj_pos, None)
				pop_these = []
				for key in new_graph["node_pos_dependencies"].keys():
					pop_this = False
					for dep_obj_pos, _ in new_graph["node_pos_dependencies"][key]:
						if dep_obj_pos == obj_pos:
							pop_this = True
						else:
							pass
					if pop_this:
						pop_these.append(key) 
					else:
						pass
				for key in pop_these:
					_ = new_graph["node_pos_dependencies"].pop(key, None)
				change_structure = True
				break
			else:
				pass

		init_x, init_y = np.where(node.state_space == 1)
		new_state[init_x[0], init_y[0]] = 0
		new_state[agent_pos[0], agent_pos[1]] = 1

		new_node = Node(new_inventory, new_state, new_graph)
		new_node.reward_so_far = node.reward_so_far + np.multiply(new_inventory - node.inventory, self.bonus_inventory).sum()
		new_node.skills_so_far = node.skills_so_far + [(option[0], option[1])]
		new_node.cost_so_far = node.cost_so_far + cost
				
		return new_node, (change_inventory or change_structure)


	def play(self, initial_config, use_graph_guide=True):
		# 
		# alpha: independent to reusable/bonus event reward ratio
		# beta: cost/reward ratio that we'll care for
		# gamma: same as RL. discount factor (used to propagate rewards through the graph)
		#
		solutions = []
		to_search = []
		# Removing direction indicator
		dir_x, dir_y = np.where(initial_config%1 == 0.5)
		initial_config[dir_x, dir_y] += 0.5
		# Configure start node
		start_node = Node(np.zeros(self.required_inventory.shape), initial_config, self.graph_guide)
		## test
		start_node.inventory[8] = 1
		self.graph_cost(start_node)
		to_search.append(start_node)
		# Graph search
		count = 0
		while len(to_search) > 0:
			count += 1
			node = to_search.pop(0)
			available_options, costs = self.get_next_options(node, use_graph_guide)
			# Running through possible options
			for option, cost in zip(available_options, costs):
				new_node, change = self.approximate_transition_step(node, option, cost, use_graph_guide)
				if not change:
					continue
				if (new_node.inventory - self.required_inventory >= 0).all():
					indices_to_remove = []
					pareto_front = True
					for i, prev_sol in enumerate(solutions):
						if (prev_sol.reward_so_far >= new_node.reward_so_far) and (prev_sol.cost_so_far <= new_node.cost_so_far):
							pareto_front = False
						elif (prev_sol.reward_so_far <= new_node.reward_so_far) and (prev_sol.cost_so_far >= new_node.cost_so_far):
							indices_to_remove.append(i)
					if pareto_front: solutions.append(new_node)
					for i, ind in enumerate(indices_to_remove):
						_ = solutions.pop(ind - i)
				# Should we further search this node. As of now, we're just saying yes to everything
				to_search.append(new_node)
				# print(count, len(solutions), len(to_search))
		return solutions


def main():
	# Initialise agent and rulebook
	system1 = System1Adapted()
	system2 = System2()
	system2.rule_dict = pickle.load(open("rule_dict.pk", "rb"))
	system3 = System3(system2.rule_dict)
	# Load demos
	for demo in [pickle.load(open("demos.pk", "rb"))[-1]]:
		# System 1, segments original demo
		demo_model = [ fullstate(s) for s in demo ]
		for state in demo_model:
			system1.next_state(state)
		segmentation_index, skill_sequence = system1.result()
		# System 2, update rules and get event/rule sequence
		num_rules_prev = len(system2.rule_dict)
		rule_sequence, reachability_set_sequence, event_position_sequence = system2.what_happened(skill_sequence, system1)
		# System 3, infers objective, generates graph guide, and outputs skill sequence for the new environment
		##
		### To do ###
		## 
		# Here, we're using the same environment, we should experiment with multiple environments
		#
		objective = system3.infer_objective(rule_sequence, reachability_set_sequence, event_position_sequence)
		new_env = pickle.load(open("env3.pk", "rb"))
		graph_guide = system3.get_dependency_graph_guide(system1.observation_function(new_env))
		possible_skill_sequences = system3.play(system1.observation_function(new_env))
		system3.reset_demo_objectives()
	#print("Final set of rules: \n\n".format())
	#for i, rule in enumerate(agent.rule_dict):
	#		print("Rule Number:{} || obj:{}\nrules:{}\nconditions:{}\n\n".format(i, rule["object"], rule["rules"], rule["conditions"]))
	import ipdb; ipdb.set_trace()
		


if __name__ == "__main__":
	main()

