import numpy as np
from segmentation_agent_pos import predict_sequence

# ------


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--phase-1', default=False, action="store_true")
parser.add_argument('--phase-2', default=False, action="store_true")
parser.add_argument('--phase-3', default=False, action="store_true")
parser.add_argument('--phase-4', default=False, action="store_true")
parser.add_argument('--all-phases', default=False, action="store_true")
args = parser.parse_args()


# ------


# The solution sketch for demos
key2sketch = { 0: [1,4], 1: [1,5], 2: [3,6], 3: [3,4], 4: [2,1,6], 5: [1,4,3,5], \
				   6: [1,5,2,4], 7: [1,5,2,5], 8: [2,1,6,7], 9: [1,5,2,4,8] }

import pickle
demos = pickle.load(open("../data_psketch/demo_dict.pk", "rb"))
maps = pickle.load(open("../data_psketch/maps.pk", "rb"))


# ------


def test(demos, key):
	correct = 0
	incorrect = 0
	for demo in demos[key]:
		try:
			Q, seg = predict_sequence(demo)
			if Q == key2sketch[key]: 
				correct+=1
			else:
				incorrect+=1
		except Exception as e:
			print("Exception occurred: {}".format(e))
			incorrect+=1
	return correct/(correct+incorrect)


def reproduce_demo(demo):
	# Determine the correct map for failure case
	map_demo = maps[19][20]
	# Determine the sketch, where the demo is failing
	sketch, _ = predict_sequence(demo)
	return map_demo, sketch


# ------


def main():

	# Phase 1: Unconscious incompetence
	if args.phase_1 or args.all_phases:
		print("Starting phase 1")
		# Check for performance and failure cases
		for key in demos.keys():
			try:
				accuracy = test(demos, key)
				if accuracy > 0.9:
					print("Success for key {}. Accuracy {}".format(key, accuracy))
				else:
					print("Failure for key {}. Accuracy {}".format(key, accuracy))
			except:
				print("Couldn't perform for key {}".format(key))
				

	# Phase 2: Conscious incompetence
	if args.phase_2 or args.all_phases:
		print("Starting phase 2")
		# Choose the correct failure case to tackle
		demo = demos[9][936]
		# Understand to reproduce it
		failure_specifications = reproduce_demo(demo)
		map_demo, sketch = failure_specifications
		# Here although we already pick the demo to be tested, 
		# this is supposed to be a part of a full process, where
		# we examine all the failure cases, and pick one thing to work on


	# Phase 3
	if args.phase_3 or args.all_phases:
		# You picked the next challenge to tackle
		# Find out environment specifications and action sequences that 
		# lead to this behaviour
		# And when it doesn't
		demo = demos[9][936]
		map_demo = maps[19][20]
		sketch = [1, 5, 2, 4, 8, (0,6)]
		import ipdb; ipdb.set_trace()


	# Phase 4
	if args.phase_4 or args.all_phases:
		pass
		# Learn the concept of inventory
		# Learn the correlation, fit all the new "rules" in to this model
		# Find out that this smoothly works with all the examples
		# Occam's razor method. Except you need to find the right framework so that this works



if __name__ == "__main__":
	main()

