import pickle
import random
import numpy as np

from craft.envs.craft_world import CraftScenario, CraftWorld
from system1 import EnvironmentHandler
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

	def infer_objective(self, rule_sequence, reachability_set_sequence, event_position_sequence, initial_config):
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
			import ipdb; ipdb.set_trace()
			transition, pre_requisites, _ =  self.rule_dict[rule[0]]
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
			inventory_used = np.where(pre_requisites[0] == 1)
			possible_parents_inventory = {}
			for obj_used in inventory_used:
				possible_parents_inventory[obj_used] = []
				for obj, possible_parent in inventory_condition_buffer:
					if obj == obj_used:
						possible_parents_inventory[obj_used].append(possible_parent)
			# Check if the objects being used are being exhausted
			inventory_used_up = np.where(transition[0][:-1] == -1)[0]
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
				# Check if the object gets exhausted when used up
				if obj in inventory_used_up:
					ind = possible_parents_inventory[obj].index(possible_parent)
					_ = possible_parents_inventory[obj].pop(ind)
			# Check if any objects are added in the inventory
			inventory_added = np.where(transition[0][:-1] == 1)[0]
			for obj in inventory_added:
				inventory_condition_buffer.append((obj, i))
			import ipdb; ipdb.set_trace()


	def get_optimal_task_sequence(self, initial_config):
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
	for demo in [pickle.load(open("demos.pk", "rb"))[-1]]:
		# Let system 1, do the work
		demo_model = [ fullstate(s) for s in demo ]
		for state in demo_model:
			system1.next_state(state)
		segmentation_index, skill_sequence = system1.result()
		# Now system 2, update rules and get result
		num_rules_prev = len(system2.rule_dict)
		rule_sequence, reachability_set_sequence, event_position_sequence = system2.what_happened(skill_sequence, system1)
		# We need to print graph here
		possible_objectives = system3.infer_objective(rule_sequence, reachability_set_sequence, event_position_sequence, demo_model[0])
	#print("Final set of rules: \n\n".format())
	#for i, rule in enumerate(agent.rule_dict):
	#		print("Rule Number:{} || obj:{}\nrules:{}\nconditions:{}\n\n".format(i, rule["object"], rule["rules"], rule["conditions"]))
	import ipdb; ipdb.set_trace()
		


if __name__ == "__main__":
	main()

