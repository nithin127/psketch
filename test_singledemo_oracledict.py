import pickle
import numpy as np
from system3 import *


# Single demonstration

demos_water = pickle.load(open("demos_water_gold.pk", "rb"))
demo = demos_water['1layer'][1]

# Test environments

test_env = pickle.load(open("maps__test.pk", "rb"))
train_env = pickle.load(open("maps__train.pk", "rb"))



# Our method (perfect rule dict)

system1 = System1Adapted()
system2 = System2()
system2.rule_dict = pickle.load(open("rule_dict.pk", "rb"))
system3 = System3(system2.rule_dict)

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

success = 0
failure = 0



#i = 2


for i, env in enumerate(train_env[21:]):
#for i, env in enumerate([train_env[20]]):
	state = env
	observable_env = system1.observation_function(fullstate(state))
	graph_guide = system3.get_dependency_graph_guide(observable_env)
	state.render()
	state.render()
	input("\n\n\n\nEnvironment number: {}\n\n\n\n\n".format(i+21))
	possible_skill_sequences = system3.play(observable_env)
	#import ipdb; ipdb.set_trace()
	for skill_params, obj in possible_skill_sequences[0].skills_so_far:
		observable_env = system1.observation_function(fullstate(state))
		pos_x, pos_y = np.where(observable_env == 1)
		action_seq = system1.use_object(observable_env, (pos_x[0], pos_y[0]), skill_params)
		for a in action_seq:
			_, state = state.step(a)
	if state.inventory[10] > 0:
		success += 1
	else:
		failure += 1
		print("Failure case number: {}".format(i))
	state.render()
	state.render()
	

