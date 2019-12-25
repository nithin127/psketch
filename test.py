import pickle
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


import ipdb; ipdb.set_trace()


for env in all_envs:
	graph_guide = system3.get_dependency_graph_guide(system1.observation_function(env))
	possible_skill_sequences = system3.play(system1.observation_function(env))




