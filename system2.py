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
		# These things can be replaced by neural networks
		self.rule_dict = {}
		self.rule_sequence = []
		self.reachability_set_sequence = []
		self.event_position_sequence = []
		self.current_inventory = np.zeros(21)
		self.rule_dict_oracle = pickle.load(open("rule_dict.pk", "rb"))
		self.craft_scenario = CraftScenario		
		self.craft_world = CraftWorld		

	def restart(self):
		self.rule_sequence = []
		self.reachability_set_sequence = []
		self.event_position_sequence = []
		self.current_inventory = np.zeros(21)


	def what_happened(self, events, system1):
		self.reachability_set_sequence += [ system1.object_reachability_set_initial ]
		# Now let's see what happened in events
		print("------------------------")
		print("   Describing events    ")
		print("------------------------")
		for ie, event in enumerate(events):
			if not event["object_before"] in self.rule_dict.keys():
				state_set = system1.environment_handler.get_full_state_set(event)
				prev_inventory_set, transitions_set = self.fully_analyse_object(state_set)
				success = self.add_to_rule_base(prev_inventory_set, transitions_set, event["object_before"])
				print("Training agent for event {}".format(event))
				if success == False:
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
			self.reachability_set_sequence += [event["new_reachable_objects"]]
			self.event_position_sequence += [event["event_location"]]
		print("------------------------")
		# Let's update the reachability graph (we don't have to)
		#self.update_graph()
		return self.rule_sequence, self.reachability_set_sequence, self.event_position_sequence


	def use_demo(self, demo, system1):
		# Let system 1, do the work
		demo_model = [ fullstate(s) for s in demo ]
		for state in demo_model:
			system1.next_state(state)
		segmentation_index, skill_sequence = system1.result()
		# Now system 2, update rules and get result
		num_rules_prev = len(self.rule_dict)
		rule_sequence, reachability_set_sequence, event_position_sequence = self.what_happened(skill_sequence, system1)
		# We need to print graph here
		print("{} new rules added".format(len(self.rule_dict) - num_rules_prev))
		system1.restart()
		self.restart()
		return rule_sequence, reachability_set_sequence, event_position_sequence


	def explore_env_specific(self, state, system1, object_sequence, print_rule_base = True):
		# Hmm, let's see
		for obj in object_sequence:
			agent_obs = system1.observation_function(fullstate(state))
			agent_pos = state.pos
			obj_pos_x, obj_pos_y = np.where(agent_obs == obj)
			obj_pos_x, obj_pos_y = obj_pos_x[0], obj_pos_y[0]
			action_sequence = system1.use_object(agent_obs, (agent_pos[0], agent_pos[1]), \
								(obj_pos_x, obj_pos_y))
			
			if action_sequence[-1] == 4:
				for a in action_sequence[:-1]:
					_, state = state.step(a)
				
				prev_inventory_set, transition_set = self.fully_analyse_object([state])
				success = self.add_to_rule_base(prev_inventory_set, transition_set, obj)
				_, state = state.step(action_sequence[-1])
				
				#import ipdb; ipdb.set_trace()

				if print_rule_base:
					for key in self.rule_dict:
						print("Object:", num_string_dict[key])
						if len(self.rule_dict[key][0].shape) == 2:
							for i in range(len(self.rule_dict[key][0])):
								print("Transition:", self.rule_dict[key][0][i].astype(np.int64), "\nPre-requisite:", self.rule_dict[key][1][i].astype(np.int64), "\n", self.rule_dict[key][2][i], "\n")
						else:
							print("Transition:", self.rule_dict[key][0].astype(np.int64), "\nPre-requisite:", self.rule_dict[key][1].astype(np.int64), "\n", self.rule_dict[key][2])
						#print("\n")


	def explore_env(self, environments, system1, num_unique_envs = 3, num_envs = 1000, max_skills_per_env = 15):
		unique_initial_environments = np.random.choice(environments, min(num_unique_envs, len(environments)), replace=False)
		env_indices = np.random.choice(len(unique_initial_environments), num_envs, replace=True)
		unique_initial_environments = [env.init() for env in unique_initial_environments]
		#state_set = {}
		correct_rules_growth = []
		compounded_rules_growth = []
		incorrect_rules_growth = []
		total_rules_growth = []
		# The thing
		for ind in env_indices:
			grid, pos = unique_initial_environments[ind].grid, unique_initial_environments[ind].pos
			state = self.craft_scenario(grid, pos, self.craft_world())
			state = state.init()
			obs_env = system1.observation_function(fullstate(state))
			# Get all the skill options beforehand
			dir_x, dir_y = np.where(obs_env%1 == 0.5)
			obs_env[dir_x[0]][dir_y[0]] += 0.5
			obs_env = obs_env.astype(np.int64)
			skill_options = np.where(obs_env > 2)
			option_ind = list(range(len(skill_options[0])))

			for _ in range(max_skills_per_env):
				# Pick skill
				agent_obs = system1.observation_function(fullstate(state))
				dir_x, dir_y = np.where(agent_obs%1 == 0.5)
				agent_obs[dir_x[0]][dir_y[0]] += 0.5
				agent_obs = agent_obs.astype(np.int64)
				agent_pos = state.pos
				option = option_ind.pop(np.random.choice(len(option_ind)))
				obj = int(agent_obs[skill_options[0][option], skill_options[1][option]])
				# Execute and gather critical states
				try:
					action_sequence = system1.use_object(agent_obs, (agent_pos[0], agent_pos[1]), \
					(skill_options[0][option], skill_options[1][option]))
				except:
					pass
				if action_sequence[-1] == 4:
					for a in action_sequence[:-1]:
						_, state = state.step(a)
					prev_inventory_set, transition_set = self.fully_analyse_object([state])
					success = self.add_to_rule_base(prev_inventory_set, transition_set, obj)
					correct_rules, compounded_rules, incorrect_rules = self.analyse_rule_base()
					correct_rules_growth.append(correct_rules)
					compounded_rules_growth.append(compounded_rules)
					incorrect_rules_growth.append(incorrect_rules)
					total_rules_growth.append(correct_rules+incorrect_rules+compounded_rules)

					print (success, (correct_rules, compounded_rules, incorrect_rules), [len(self.rule_dict[key][0]) for key in self.rule_dict.keys()])
						
					_, state = state.step(action_sequence[-1])
				else:
					pass
				#if obj in state_set.keys():
				#	state_set[obj].append(state)
				#else:
				#	state_set[obj] = [state]

				# Break condition
				if len(option_ind) == 0:
					break

		return correct_rules_growth, compounded_rules_growth, incorrect_rules_growth, total_rules_growth 

		# Now plot

		"""
		num_new_rules = 0
		for obj in state_set.keys():
			prev_inventory_set, transition_set = self.fully_analyse_object(state_set[obj])
			success = self.add_to_rule_base(prev_inventory_set, transition_set, obj)
			if success is not False:
				num_new_rules += success
		
		print("{} new rules added".format(num_rules_prev))
		"""

	def fully_analyse_object(self, state_set):
		condition_set = np.empty((0, 21))
		transition_set = np.empty((0, 22))

		for i, ss in enumerate(state_set):
			_, sss = ss.step(4)
			# object_in_front_difference should only be -1 or 0, or it is disaster
			pos = ss.pos
			dirc = ss.dir
			if dirc == 0:
				dir_pos = (pos[0], pos[1] - 1)
			elif dirc == 1:
				dir_pos = (pos[0], pos[1] + 1)
			elif dirc == 2:
				dir_pos = (pos[0] - 1, pos[1])
			elif dirc == 3:
				dir_pos = (pos[0] + 1, pos[1])
			object_in_front_difference = np.clip(sss.grid[dir_pos[0], dir_pos[1]].argmax() - ss.grid[dir_pos[0], dir_pos[1]].argmax(), -1, 1)
			transition = np.expand_dims(np.append(sss.inventory - ss.inventory, object_in_front_difference), axis = 0)
			condition_set = np.append(condition_set, np.expand_dims(ss.inventory, axis = 0), axis = 0)
			transition_set = np.append(transition_set, transition, axis = 0)
		
		return condition_set, transition_set


	def add_to_rule_base(self, condition_set, transition_set, rule_object):
		if rule_object in self.rule_dict.keys():
			prev_transitions = self.rule_dict[rule_object][0]
			prev_conditions = self.rule_dict[rule_object][1]
			transition_set = np.append(transition_set, prev_transitions, axis = 0)
			condition_set = np.append(condition_set, prev_conditions, axis = 0)

		unique_transitions = np.unique(transition_set, axis = 0)
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
				tr_indices = np.where((unique_transitions[ind] == transition_set).all(axis=1))[0]
				prev_inventory_subset = np.empty((0, 21))
				for tr_ind in tr_indices:
					prev_inventory_subset  = np.append(prev_inventory_subset, np.expand_dims(condition_set[tr_ind], axis=0), axis = 0)
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
							format(num_string_dict[rule_object], text_used_up))
				else:
					desc_set.append("Used {} to make {} at {}".\
						format(text_used_up, text_gathered, num_string_dict[rule_object]))


		try:
			if core_transitions.shape[1] == 0:
				return False
			self.rule_dict[rule_object] = (core_transitions.T, pre_requisite_set, desc_set)
			return True
		except:
			return False
		

	def analyse_rule_base(self):
		correct_rules = 0
		compounded_rules = 0
		incorrect_rules = 0

		gathered_keys = set(self.rule_dict.keys())
		gt_keys = set(self.rule_dict_oracle.keys())
		for key in gathered_keys:
			tr_gathered = self.rule_dict[key][0]
			pre_gathered = self.rule_dict[key][1]
			# Check if transitions are correct, or compounded
			for transition, pre_requisite in zip(tr_gathered, pre_gathered):
				gt_transitions = self.rule_dict_oracle[key][0]
				gt_prerequisite = self.rule_dict_oracle[key][1]
				matrix = np.append(gt_transitions, np.expand_dims(transition, axis =0), axis = 0)
				if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
					# This is a new rule, not present in the oracle
					incorrect_rules += 1
				else:
					'''
					# Get coefficient and check pre-requisite accordingly				
					coeff = np.linalg.lstsq(gt_transitions, np.expand_dims(transition, axis =0), rcond=None)
					if not np.allclose(gt_prerequisite*coeff - pre_gathered):
						incorrect_rules += 1
					else:
						import ipdb; ipdb.set_trace()
						if coeff.sum() == 1:
							correct_rules += 1
						else:
							compounded_rules += 1
					'''
					# Shortcut
					compounded = True
					for g_tr, g_pre in zip(gt_transitions, gt_prerequisite):
						if (g_tr - transition == 0).all():
							if (g_pre - pre_requisite == 0).all():
								correct_rules += 1
							else:
								incorrect_rules += 1
							compounded = False
							break

					if compounded:
						compounded_rules += 1

		return correct_rules, compounded_rules, incorrect_rules


def get_mean_std(plot_set):
	max_points = 1e10
	for x in plot_set:
		max_points = min(len(x), max_points)

	y = np.zeros((0,max_points))


	for x in plot_set:
		y = np.append(y, np.expand_dims(x[:max_points], axis=0), axis=0)


	error = y.std(axis=0)
	y = y.mean(axis=0)

	return y, error, max_points



def main():
	# Initialise agent and rulebook
	system1 = System1Adapted()
	system2 = System2()

	# Input playground environment, and link systems
	environment_handler = EnvironmentHandler()
	system1.environment_handler = environment_handler
	
	# Prepare input
	#input_system2 = [pickle.load(open("demos.pk", "rb"))[-1], "demo"]
	#input_system2 = [pickle.load(open("custom_maps.pk", "rb")), "env"]
	input_system2 = [pickle.load(open("custom_maps.pk", "rb")), "env_specific"]
	
	# Feed to system 2
	if input_system2[1] == "demo":
		rule_sequence, reachability_set_sequence, event_position_sequence = system2.use_demo(input_system2[0], system1)

	elif input_system2[1] == "env":
		correct_set = []
		compounded_set = []
		incorrect_set = []
		total_set = []
		for _ in range(2):
			correct, compounded, incorrect, total  = system2.explore_env(input_system2[0], system1, num_unique_envs = 3, num_envs = 1000, max_skills_per_env = 20)
			correct_set.append(correct)
			compounded_set.append(compounded)
			incorrect_set.append(incorrect)
			total_set.append(total)

		correct_mean, correct_std, max_points = get_mean_std(correct_set)
		compounded_mean, compounded_std, max_points = get_mean_std(compounded_set)
		incorrect_mean, incorrect_std, max_points = get_mean_std(incorrect_set)
		total_mean, total_std, max_points = get_mean_std(total_set)

		from matplotlib import pyplot as plt
		fig, ax = plt.subplots()

		x = np.linspace(0, max_points-1, max_points)
		y_oracle = np.asarray([16]*max_points)
		ax.fill_between(x, correct_mean-correct_std, correct_mean+correct_std, color='g', alpha=0.3)
		ax.fill_between(x, compounded_mean-compounded_std, compounded_mean+compounded_std, color='b', alpha=0.3)
		ax.fill_between(x, (incorrect_mean-incorrect_std), (incorrect_mean+incorrect_std), color='r', alpha=0.3)
		ax.plot(x, y_oracle, 'k--', label='Baseline')
		ax.plot(x, correct_mean, 'g-', label='Avg. correct rules')
		ax.plot(x, compounded_mean, 'b-', label='Avg. compounded rules')
		ax.plot(x, incorrect_mean, 'r-', label='Avg. incorrect rules')

		legend = ax.legend(loc='lower right', shadow=False, fontsize='x-small')
		plt.title("1000 environments, 20 skills per environment")

		# Put a nicer background color on the legend.
		# legend.get_frame().set_facecolor('C0')
		plt.show()


	elif input_system2[1] == "env_specific":
		inventory = np.zeros(21)
		#inventory[8] = 1
		#inventory[9] = 1
		system2.explore_env_specific(np.random.choice(input_system2[0]).init(inventory), system1, [12,7,8])
		print("\n\n\n\n\n")
	
	# Print final rule dict	
	#for key in self.rule_dict:
	#	print("Object:", num_string_dict[key])
	#	if len(self.rule_dict[key][0].shape) == 2:
	#		for i in range(len(self.rule_dict[key][0])):
	#			print("Transition:", self.rule_dict[key][0][i].astype(np.int64), "\nPre-requisite:", self.rule_dict[key][1][i].astype(np.int64), "\n")
	#	else:
	#		print("Transition:", self.rule_dict[key][0].astype(np.int64), "\nPre-requisite:", self.rule_dict[key][1].astype(np.int64), "\n")
	#	print("\n")

	import ipdb; ipdb.set_trace()
	pickle.dump(system2.rule_dict, open("rule_dict.pk", "wb"))



if __name__ == "__main__":
	main()

