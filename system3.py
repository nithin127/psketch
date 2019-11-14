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


# --------------------------------------- Agent Function -------------------------------------- #		

class System3():
	def __init__(self, rule_dict):
		self.rule_dict = rule_dict

	def infer_objective(self, rule_sequence, reachability_set_sequence, initial_config):
		import ipdb; ipdb.set_trace()
		# This function outputs a reward vector in terms of events, and a reward vector in terms of the minimal 
		# possible final inventory. Each  
		#
		# There are independent events: ones which are not pre-requisites for any other event ( + 2 reward )
		# There are some events which would be independent if we're not taking reachability into account ( + 0.5 reward )
		# There are reusable events: ones which can repeatedly act as a pre-requisite for multiple event ( + 1 reward )
		# Then there is a minimal possible final inventory objective
		#
		# Let's find 
		pass

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
		rule_sequence, reachability_set_sequence = system2.what_happened(skill_sequence)
		# We need to print graph here
		system3.infer_objective(rule_sequence, reachability_set_sequence, demo_model[0])
	#print("Final set of rules: \n\n".format())
	#for i, rule in enumerate(agent.rule_dict):
	#		print("Rule Number:{} || obj:{}\nrules:{}\nconditions:{}\n\n".format(i, rule["object"], rule["rules"], rule["conditions"]))
	import ipdb; ipdb.set_trace()
		


if __name__ == "__main__":
	main()

