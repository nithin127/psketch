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

demo_model = [ system1.observation_function(fullstate(s)) for s in demo ]
prev_state = demo_model[0]
action_seq = []

for state in demo_model[1:]:
	px, py = np.where(prev_state == 1)
	cx, cy = np.where(state == 1)
	if cy - py == 1:
		assert px == cx
		action_seq.append(1)
	elif cy - py == -1:
		assert px == cx
		action_seq.append(0)
	else:
		if cx - px == 1:
			assert cy == py
			action_seq.append(3)
		elif cx - px == -1:
			assert cy == py
			action_seq.append(2)
		elif cy == py and cx == px:
			pdx, pdy = np.where(prev_state % 1 == 0.5)
			cdx, cdy = np.where(state % 1 == 0.5)
			if pdx == cdx and pdy == cdy:
				action_seq.append(4)
			else:
				# Here we go again
				if cdy - cy == 1:
					assert cx == cdx
					action_seq.append(1)
				elif cdy - cy == -1:
					assert cx == cdx
					action_seq.append(0)
				else:
					if cdx - cx == 1:
						assert cdy == cy
						action_seq.append(3)
					elif cdx - cx == -1:
						assert cdy == cy
						action_seq.append(2)
	# Now that we have the direction
	prev_state = state


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
		for action in action_seq:
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
