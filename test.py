import pickle
import numpy as np
from system2 import System1Adapted, System2
from system3 import System3, fullstate


# Single demonstration

demos_water = pickle.load(open("demos_water_gold.pk", "rb"))
demo = demos_water['1layer'][1]

# Test environments

all_envs = []

all_envs += pickle.load(open("maps_1layer.pk", "rb"))
all_envs +=  pickle.load(open("maps_2layer.pk", "rb"))
all_envs +=  pickle.load(open("maps_3layer.pk", "rb"))
all_envs +=  pickle.load(open("maps_gem_gold.pk", "rb"))
all_envs +=  pickle.load(open("maps_grass_gold.pk", "rb"))
all_envs +=  pickle.load(open("maps_iron_gold.pk", "rb"))
all_envs +=  pickle.load(open("maps_wood_gold.pk", "rb"))
all_envs +=  pickle.load(open("maps_stone_gold.pk", "rb"))
all_envs +=  pickle.load(open("maps_water_gold.pk", "rb"))


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


for i, env in enumerate([all_envs[5]]):
	state = env
	observable_env = system1.observation_function(fullstate(state))
	graph_guide = system3.get_dependency_graph_guide(observable_env)
	#import ipdb; ipdb.set_trace()
	print(i)
	state.render()
	state.render()
	possible_skill_sequences = system3.play(observable_env)
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
	input()

