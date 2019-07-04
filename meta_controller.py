import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os, time
import numpy as np
from logger import Logger

# Network architecture:

''' 
Take in; (state, goal) --> predict the next subtasks, and "How many subtasks you said... Hmm good question"
Look at NLP literature
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
			out.extend([0 for _ in range(6-len(out))])
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


class PredictorRNN(nn.Module):
	"""docstring for PredictorRNN"""
	def __init__(self, input_size, hidden_size, num_classes):
		super(PredictorRNN, self).__init__()
		self.hidden_size = hidden_size
		self.gru = nn.GRU(input_size, hidden_size)
		self.out = nn.Linear(hidden_size, num_classes)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, inp, hidden):
		output, hidden = self.gru(torch.tensor(inp).unsqueeze(0), hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self, minibatch_size=16):
		return torch.zeros(1, minibatch_size, int(self.hidden_size), device=device)


def trainIters(decoder, dataset, n_iters, all_losses, logger, minibatch_size=16, print_every=100, 
	plot_every=10, learning_rate=0.01, test=False):
	start = time.time()
	print_loss_total = 0  	# Reset every print_every
	plot_loss_total = 0 	# Reset every plot_every

	decoder_optimizer = optim.Adam(decoder.parameters())
	for i in range(n_iters):
		input_tensors, (target_tensors, flags) = dataset.sample(minibatch_size)
		criterion = F.nll_loss

		loss = train(input_tensors, target_tensors, flags, decoder, decoder_optimizer, minibatch_size, criterion)
		all_losses.append(loss)
		print_loss_total += loss
		plot_loss_total += loss
		# Printing and plotting functionality to be added soon
		if i % plot_every == 1:
			logger.log_scalar_avg("loss/update", all_losses, 10, len(all_losses))
		if i % print_every == 1:
			print("Training Loss at {}: {}".format(i, print_loss_total/i))

	return all_losses


def get_one_hot(vector, num_classes):
	yo = torch.zeros((len(vector), num_classes), device=device)
	for i, t in enumerate(vector):
		yo[i][t] = 1
	return yo


def train(input_tensor, target_tensor, flags, decoder, decoder_optimizer, minibatch_size, criterion):
	decoder_optimizer.zero_grad()
	inp = get_one_hot(minibatch_size*[10], 11)

	loss = torch.zeros(1, dtype=torch.float, device=device)
	hidden = decoder.initHidden(minibatch_size)

	for t, f in zip(target_tensor.transpose(0,1), flags.transpose(0,1)):
		out, hidden = decoder(torch.cat((input_tensor, inp), dim=1), hidden)
		loss_i = criterion(out, t, reduction='none')*f
		loss += loss_i.sum()
		inp = get_one_hot(t, 11)

	loss = loss/flags.sum()
	loss.backward()
	decoder_optimizer.step()
	return loss.item()


def main():
	if os.path.exists("experiments/hrl"):
		import shutil
		shutil.rmtree("experiments/hrl")
	logger = Logger("experiments/hrl")

	decoder = PredictorRNN(1098, 512, 11)
	if torch.cuda.is_available():
		decoder.cuda()
	data = dataset("dataset_hrl.pk")
	all_losses = []
	for _ in range(1000):
		print("Epoch: {}".format(_))
		all_losses = trainIters(decoder, data, 100, all_losses, logger)
		logger.log_scalar_avg("loss/epoch", all_losses, 10, _)
		if _ % 30 == 1:
			torch.save({"epoch": _, "all_losses": all_losses,
				"decoder": decoder.state_dict()}, "experiments/hrl/checkpoint.pt")


if __name__ == "__main__":
	main()

