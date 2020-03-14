import pickle
import numpy as np
from system1 import *


# Single demonstration

demos_water = pickle.load(open("demos_water_gold.pk", "rb"))
demo = demos_water['1layer'][1]


# Test environments

test_env = pickle.load(open("maps__test.pk", "rb"))
train_env = pickle.load(open("maps__train.pk", "rb"))


# Our method (using demo)

system1 = System1()

demo_model = [ fullstate(s) for s in demo ]
for state in demo_model:
	system1.next_state(state)
segmentation_index, skill_sequence = system1.result()
		

# Just replicating the skill sequence

success = 0
success_cases = []
failure = 0
failure_cases = []


for i, env in enumerate(train_env):
#for i, env in enumerate(test_env):
	state = env
	observable_env = system1.observation_function(fullstate(state))
	state.render()
	state.render()
	input("\n\n\n\nEnvironment number: {}\n\n\n\n\n".format(i))
	try:
		for event in skill_sequence:
			observable_env = system1.observation_function(fullstate(state))
			pos_x, pos_y = np.where(observable_env == 1)
			done = False
			possible_objects = np.where(observable_env == event["object_before"])
			for skill_param_x, skill_param_y in zip(possible_objects[0], possible_objects[1]):
				if done:
					break
				try:
					action_seq = system1.use_object(observable_env, (pos_x[0], pos_y[0]), (skill_param_x, skill_param_y))
					if len(action_seq) > 0 and action_seq[-1] == 4:
						done = True
						for a in action_seq:
							_, state = state.step(a)
						break
				except:
					pass
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
