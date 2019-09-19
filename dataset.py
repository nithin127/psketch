## Plan of action

# First create the environments. Save
# Then use skills randomly in different environments, file these demos. Save ... with labels
# Use these labelled demos to come up with skill discriminators. Save model

import os, pickle
import numpy as np
from craft.envs.craft_world import CraftScenario, CraftWorld





class dataset():
	def __init__(self, dataset_name = "dataset_hrl.pk"):
		# Dataset = create_dataset()
		# pickle.dump(Dataset, open("dataset_hrl.pk", "wb"))
		import pickle
		if os.path.exists(dataset_name):
			self.data = pickle.load(open(dataset_name, "rb"))
		else:
			from create_dataset import create_dataset
			self.data = create_dataset()
			pickle.dump(self.data, open(dataset_name, "wb"))

		self.data = self.data
		self.index = 0
		self.dataset_size = len(self.data)

	def sample(self, batch_size=16):
		inps = []
		outs = []
		flags = []

		if batch_size > self.dataset_size:
			# raise("Bro, check batch_size... its too big")
			inps_, outs_ = self.sample(self.dataset_size)
			inps__, outs__ = self.sample(batch_size - self.dataset_size)
			return inps_ + inps__, outs_ + outs__

		if self.index + batch_size >= self.dataset_size - 1:
			self.randomize()

		for data in self.data[self.index: self.index + batch_size]:
			goal = data[1] - 10
			out = data[2].copy()
			flag = torch.cat((torch.ones(len(out)+1, device=device), torch.zeros(6-len(out)-1, device=device)))
			out.extend([0 for _ in range(6 - len(out))])
			inp = np.zeros(11)
			inp[goal] = 1
			inp = torch.tensor(np.append(inp, data[0].features()), dtype=torch.float, device = device)
			inps.append(inp)
			outs.append(torch.tensor(out, device = device))
			flags.append(flag)
		self.index += batch_size
		return torch.stack(inps), (torch.stack(outs), torch.stack(flags))

	def randomize(self):
		self.index = 0
		np.random.shuffle(self.data)



def create_maps(num_maps = 1000):
	# create_maps
	cw = CraftWorld()
	map_set = []
	while len(map_set) < num_maps:
		#if len(map_set) % 25 == 0:
		#	print(len(map_set))
		goal = np.random.randint(14) + 7
		scenario = cw.sample_scenario_with_goal(goal)
		map_i = scenario.init()
		append = True
		for map_j in map_set:
			if (map_i.grid == map_j.grid).all():
				append = False
		if append:
			map_set.append(map_i)
	return map_set



def main():
	
	if not os.path.exists('all_maps.pk'):
		map_set = create_maps()
		pickle.dump(map_set, "all_maps.pk")
	else:
		map_set = pickle.load(open("all_maps.pk", "rb"))
	# Divide in train and test
	train_maps = map_set[:800]
	test_maps = map_set[800:]
	import ipdb; ipdb.set_trace()
	# create demos
	train_demos = create_demos(train_maps)
	test_demos = create_demos(test_maps)



if __name__ == "__main__":
	main()



## Comments

# There are no negative examples here, and we're eliminating the other "skills"
# Here the skills are not parameterised, except -- yeah, the location of the skill is kind of parameterised
# Get(wood, 3, 2). There is only one skill that is being parameterised


# Discriminator -- segment first, and another layer that predicts the skill number and the location parameters
# This should be relatively straightforward

# Test of generalisation, new skills, new things. 