import pickle
import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from main_one import *
from main_two import *
from craft.envs import CraftEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_map = {
	10: [1,2,6,7,0],
	11: [1,5,2,4,8,0],
	12: [1,4,0],
	13: [1,5,0],
	14: [1,5,2,4,0], 
	15: [3,4,0],
	16: [1,4,3,5,0],
	17: [1,5,2,5,0],
	18: [3,6,0],
	19: [2,1,6,0]
}

index_goal = {
	10: "gold",
	11: "gem",
	12: "plank",
	13: "stick",
	14: "axe",
	15: "rope",
	16: "bed",
	17: "shears",
	18: "cloth",
	19: "bridge",
}

###------------------------------------- Arguments ----------------------------------------###


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--visualise-example', default=False, action="store_true")
parser.add_argument('--goal', default=12, type=int, help='The goal value: int 10 to 19')
parser.add_argument('--exp_dir', default='experiments/hrl/checkpoint.pt')
parser.add_argument('--minibatch_size', default=1, help='minibatch size')
parser.add_argument('--num_classes', default=11, help='num_classes')
args = parser.parse_args()


###---------------------------------- Main Functions --------------------------------------###


def get_one_hot(vector, num_classes):
	yo = torch.zeros((len(vector), num_classes), device=device)
	for i, t in enumerate(vector):
		yo[i][t] = 1
	return yo

def get_decoder_input(input_state, goal):
	inp = get_one_hot([goal-10], args.num_classes)
	return torch.tensor(np.append(inp, input_state.features()), dtype=torch.float, device = device)


def get_sequence_all(input_state, decoder, target_sequence, goal, criterion):
	# Computes the whole sub-sequence, as the way it is trained
	seq = []
	input_tensor = get_decoder_input(input_state, goal)
	loss = torch.zeros(1, dtype=torch.float, device=device)
	# Start decoding
	inp = get_one_hot(1*[10], args.num_classes)
	hidden = decoder.initHidden(1)
	for tr in target_sequence:
		out, hidden = decoder(torch.cat((input_tensor.unsqueeze(0), inp), dim=1), hidden)
		loss += criterion(out, torch.tensor([tr], device=device))
		t = torch.argmax(out).item()
		seq.append(t)
		inp = get_one_hot(1*[t], args.num_classes)
	return seq, loss.item()/len(target_sequence)


def get_reward(input_state, decoder, goal, max_iter = 20):
	# This applies the meta controller after every sub-policy execution
	# and uses the forward dynamics model
	i = 0
	reward = 0
	subpolicydidntwork = 0
	inp = get_one_hot(1*[10], args.num_classes)
	hidden = decoder.initHidden(1)
	while i < max_iter:
		out, hidden = decoder(torch.cat((get_decoder_input(input_state, goal).unsqueeze(0), inp), dim=1), hidden)
		sub_policy = torch.argmax(out).item()
		# print(sub_policy, i)
		if sub_policy == 0:
			break
		try:
			input_state = sample_sequence([sub_policy], input_state)[-1][-1]
		except:
			subpolicydidntwork += 1
			pass
		i += 1
		if input_state.inventory[goal] > 0:
			reward += 1
			break
	if subpolicydidntwork > 0:
		print("subpolicydidntwork: {}\n".format(subpolicydidntwork))
	return reward


def analyse(test_maps, decoder, interactive = False):
	# Prepare the test dataset
	test_goals = []
	test_states = []
	for key in test_maps.keys():
		for m in test_maps[key]:
			test_goals.append(key)
			test_states.append(m)

	# Test error
	accuracy = {}
	loss = {}
	reward = {}
	count = {}

	for g in range(10,20):
		accuracy[g] = 0
		loss[g] = 0
		reward[g] = 0
		count[g] = 0

	for i, (s, g) in enumerate(zip(test_states, test_goals)):
		# print(i)
		count[g] += 1
		reward[g] += get_reward(s, decoder, g)
		tar_seq = target_map[g]
		#import ipdb; ipdb.set_trace()
		seq, ls = get_sequence_all(s, decoder, tar_seq, g, F.nll_loss)
		loss[g] += ls
		# Loss computation is not accurate
		if (len(tar_seq) == len(seq)) and (ls == 0):
			accuracy[g] += 1

	if interactive:
		import ipdb; ipdb.set_trace()

	return accuracy, loss, reward, count


def print_stats(stats, key="Training maps"):
	print(key + ": Goal-wise Stats")
	fin_stats = np.array([0, 0, 0], dtype=np.float64)
	for k in range(10,20):
		curr_stats = np.array([stats[0][k]/stats[3][k], stats[1][k]/stats[3][k], stats[2][k]/stats[3][k]])
		print("{}:== ac: {}, ls: {}, r: {}".format(k, *curr_stats))
		fin_stats += curr_stats
	print(key + ": Overall Stats")
	print("\tAccuracy:{}\n\tLoss:{}\n\tReward:{}".format(*fin_stats/10))
	return None


###----------------------------- Main -------------------------------### 


def main():
	print(args)
	skill_dict = pickle.load(open("skill_dict.pk", "rb"))
	trajectory = skill_dict[2][0][-1]
	print("Visualising an example of skill {}".format(2))
	for state in trajectory:
		state.render()
		input("?")

	decoder = PredictorRNN(1098, 512, args.num_classes)
	if torch.cuda.is_available():
		decoder.cuda()
	checkpoint = torch.load(args.exp_dir, map_location=lambda storage, loc: storage)
	decoder.load_state_dict(checkpoint['decoder'])

	# Create a bunch of states
	test_maps = pickle.load(open("test_maps.pk", "rb"))
	train_maps = pickle.load(open("train_maps.pk", "rb"))

	# Prepare the train dataset
	train_stats = analyse(train_maps, decoder, False)
	print_stats(train_stats, "Training maps")
	test_stats = analyse(test_maps, decoder, False)
	print_stats(test_stats, "Testing maps")
	
	import ipdb; ipdb.set_trace()
	# return


if __name__ == "__main__":
	main()
