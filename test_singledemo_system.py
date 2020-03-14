import pickle
import numpy as np
from system3 import *


# Single demonstration

demos_water = pickle.load(open("demos_water_gold.pk", "rb"))
demo = demos_water['1layer'][1]


# Test environments

test_env = pickle.load(open("maps__test.pk", "rb"))
train_env = pickle.load(open("maps__train.pk", "rb"))


# Our method (using demo)

system1 = System1Adapted()
environment_handler = EnvironmentHandler()
system1.environment_handler = environment_handler

system2 = System2()


# Dict type

dict_type = "demo"

if dict_type == "oracle":
	system2.rule_dict = system2.rule_dict_oracle

elif dict_type == "demo":
	demo_type_string = np.random.choice(["1layer", "2layer", "3layer", "gem_gold", "grass_gold", "iron_gold", "stone_gold", "water_gold", "wood_gold"])
	demos_rule_dict = pickle.load(open("demos_" + demo_type_string + ".pk", "rb"))
	demo_rule_dict = np.random.choice(demos_rule_dict['1layer'])
	rule_sequence, reachability_set_sequence, event_position_sequence = system2.use_demo(demo_rule_dict, system1)

elif dict_type == "demo_explore":
	demo_type_string = np.random.choice(["1layer", "2layer", "3layer", "gem_gold", "grass_gold", "iron_gold", "stone_gold", "water_gold", "wood_gold"])
	demos_rule_dict = pickle.load(open("demos_" + demo_type_string + ".pk", "rb"))
	demo_rule_dict = np.random.choice(demos_rule_dict['1layer'])
	rule_sequence, reachability_set_sequence, event_position_sequence = system2.use_demo(demo_rule_dict, system1)

else:
	pass


# Add random exploration here
system3 = System3(system2.rule_dict)


# System 3, infers objective, generates graph guide, and outputs skill sequence for the new environment

rule_sequence, reachability_set_sequence, event_position_sequence = system2.use_demo(demo, system1)
objective = system3.infer_objective(rule_sequence, reachability_set_sequence, event_position_sequence)


success = 0
success_cases = []
failure = 0
failure_cases = []


for i, env in enumerate(train_env):
#for i, env in enumerate(test_env):
	state = env
	observable_env = system1.observation_function(fullstate(state))
	try:
		graph_guide = system3.get_dependency_graph_guide(observable_env)
	except:
		failure += 1
		failure_cases.append(i)
		continue
	state.render()
	state.render()
	print("\n\n\n\nEnvironment number: {}\n\n\n\n\n".format(i))
	possible_skill_sequences = system3.play(observable_env)
	import ipdb; ipdb.set_trace()
	try:
		for skill_params, obj in possible_skill_sequences[0].skills_so_far:
			observable_env = system1.observation_function(fullstate(state))
			pos_x, pos_y = np.where(observable_env == 1)
			action_seq = system1.use_object(observable_env, (pos_x[0], pos_y[0]), skill_params)
			for a in action_seq:
				_, state = state.step(a)
		if state.inventory[10] > 0:
			success += 1
			success_cases.append(i)
		else:
			failure += 1
			failure_cases.append(i)
	except:
		failure += 1
		failure_cases.append(i)
	state.render()
	state.render()
	
print("Success:{}, Failure:{}".format(success, failure))
import ipdb; ipdb.set_trace()
