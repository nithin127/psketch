import torch
import torch.nn as nn

class EmbeddingRNN(nn.Module):
	"""docstring for EmbeddingRNN"""
	def __init__(self, input_size, hidden_size, final_size, device, use_features = False):
		super(EmbeddingRNN, self).__init__()
		self.use_features = use_features
		self.hidden_size = hidden_size
		self.device = device
		# Create embedding
		if not self.use_features:
			self.conv_reduce = nn.Sequential(
								nn.Conv2d(12,3,3), nn.ReLU(),
								nn.Conv2d(3,1,3), nn.ReLU(),
								)
			self.embed = nn.Sequential(
							nn.Linear(8*8,input_size), nn.ReLU()
							)
		# Recurrence
		self.gru = nn.GRU(input_size, hidden_size)
		self.out1 = nn.Sequential(
						nn.Linear(hidden_size, int(hidden_size/2)), nn.ReLU(),
						nn.Linear(int(hidden_size/2), int(hidden_size/4)), nn.ReLU(),
						nn.Linear(int(hidden_size/4), 2))
		self.out2 = nn.Sequential(
						nn.Linear(hidden_size, int(hidden_size/2)), nn.ReLU(),
						nn.Linear(int(hidden_size/2), int(hidden_size/4)), nn.ReLU(),
						nn.Linear(int(hidden_size/4), final_size))
		self.loss_fn = nn.CrossEntropyLoss(reduction="none")

	def forward(self, inps, hidden):
		if not self.use_features:
			inps = torch.stack([ self.embed(self.conv_reduce(inp).view(-1,8*8)) for inp in inps ])
		output, _ = self.gru(inps, hidden)
		out1 = self.out1(output)
		out2 = self.out2(output)
		return [out1, out2], _

	def initHidden(self, minibatch_size=1):
		return torch.zeros(1, minibatch_size, int(self.hidden_size), device=self.device)

	def get_loss(self, inps, targets1, targets2, flags1, flags2):
		# Convert to numpy
		loss1 = torch.zeros(1)
		loss2 = torch.zeros(1)
		inps = torch.from_numpy(inps).float()
		targets1 = torch.from_numpy(targets1)
		targets2 = torch.from_numpy(targets2)
		flags1 = torch.from_numpy(flags1).float()
		flags2 = torch.from_numpy(flags2).float()
		# Cuda it up
		if torch.cuda.is_available():
			loss1 = loss1.cuda()
			loss2 = loss2.cuda()
			inps = inps.cuda()
			targets1 = targets1.cuda()
			targets2 = targets2.cuda()
			flags1 = flags1.cuda()
			flags2 = flags2.cuda()
		hidden = self.initHidden(inps.shape[1])
		[out1, out2], _ = self.forward(inps, hidden)
		# Loss function
		# Constructing loss, sequentially
		seq_len = inps.shape[0]
		for i in range(seq_len):
			loss_i = self.loss_fn(out1[i], targets1[:,i])*flags1[:,i]
			if flags1[:,i].sum() == 0:
				loss1+= 0
			else:
				loss1+= loss_i.sum()
		for i in range(seq_len):
			loss_i = self.loss_fn(out2[i], targets2[:,i])*flags2[:,i]
			if flags2[:,i].sum() == 0:
				loss2+= 0
			else:
				loss2+= loss_i.sum()
		return loss1/flags1.sum(), loss2/flags2.sum() # We can add hyperparameters here

	def get_prediction(self, demo):
		# Not ready yet
		prediction = []
		indices = []
		hidden = self.initHidden(1)
		for ind, state in enumerate(demo):
			s_grid = fullstate(state)
			[out1, out2], hidden = self.forward(torch.from_numpy(s_grid).unsqueeze(0).unsqueeze(0).float(), hidden)
			if out1.argmax().item() == 1:
				prediction.append(out2.argmax().item())
				indices.append(ind)
				hidden = self.initHidden(1)			
		return prediction, indices





class EmbeddingCNN(nn.Module):
	"""docstring for EmbeddingRNN"""
	def __init__(self, input_size, hidden_size, final_size, device, use_features = False):
		super(EmbeddingCNN, self).__init__()
		self.use_features = use_features
		self.hidden_size = hidden_size
		self.device = device
		# Create embedding
		if not self.use_features:
			self.conv_reduce = nn.Sequential(
								nn.Conv2d(12,3,3), nn.ReLU(),
								nn.Conv2d(3,1,3), nn.ReLU(),
								)
			self.embed = nn.Sequential(
							nn.Linear(8*8,input_size), nn.ReLU()
							)
		# Recurrence
		self.pipe = nn.Linear(input_size, hidden_size)
		self.out1 = nn.Sequential(
						nn.Linear(hidden_size, int(hidden_size/2)), nn.ReLU(),
						nn.Linear(int(hidden_size/2), int(hidden_size/4)), nn.ReLU(),
						nn.Linear(int(hidden_size/4), 2))
		self.out2 = nn.Sequential(
						nn.Linear(hidden_size, int(hidden_size/2)), nn.ReLU(),
						nn.Linear(int(hidden_size/2), int(hidden_size/4)), nn.ReLU(),
						nn.Linear(int(hidden_size/4), final_size))
		self.loss_fn = nn.CrossEntropyLoss(reduction="none")

	def forward(self, inps):
		if not self.use_features:
			inps = torch.stack([ self.embed(self.conv_reduce(inp).view(-1,8*8)) for inp in inps ])
		output = nn.ReLU()(self.pipe(inps))
		out1 = self.out1(output)
		out2 = self.out2(output)
		return [out1, out2]

	def get_loss(self, inps, targets1, targets2, flags1, flags2):
		# Convert to numpy
		loss1 = torch.zeros(1)
		loss2 = torch.zeros(1)
		inps = torch.from_numpy(inps).float()
		targets1 = torch.from_numpy(targets1)
		targets2 = torch.from_numpy(targets2)
		flags1 = torch.from_numpy(flags1).float()
		flags2 = torch.from_numpy(flags2).float()
		# Cuda it up
		if torch.cuda.is_available():
			loss1 = loss1.cuda()
			loss2 = loss2.cuda()
			inps = inps.cuda()
			targets1 = targets1.cuda()
			targets2 = targets2.cuda()
			flags1 = flags1.cuda()
			flags2 = flags2.cuda()
		[out1, out2] = self.forward(inps)
		# Loss function
		# Constructing loss, sequentially
		seq_len = inps.shape[0]
		for i in range(seq_len):
			loss_i = self.loss_fn(out1[i], targets1[:,i])*flags1[:,i]
			if flags1[:,i].sum() == 0:
				loss1+= 0
			else:
				loss1+= loss_i.sum()
		for i in range(seq_len):
			loss_i = self.loss_fn(out2[i], targets2[:,i])*flags2[:,i]
			if flags2[:,i].sum() == 0:
				loss2+= 0
			else:
				loss2+= loss_i.sum()
		return loss1/flags1.sum(), loss2/flags2.sum() # We can add hyperparameters here

	def get_prediction(self, demo):
		# Not ready yet
		prediction = []
		indices = []
		hidden = self.initHidden(1)
		for ind, state in enumerate(demo):
			s_grid = fullstate(state)
			[out1, out2], hidden = self.forward(torch.from_numpy(s_grid).unsqueeze(0).unsqueeze(0).float(), hidden)
			if out1.argmax().item() == 1:
				prediction.append(out2.argmax().item())
				indices.append(ind)
				hidden = self.initHidden(1)			
		return prediction, indices
