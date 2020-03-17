import os, pickle
import numpy as np
from system3 import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Defining model for BC

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(1, 3, 5)
        self.conv12 = nn.Conv2d(3, 1, 5)
        self.conv_shape1 = 4

        #self.conv21 = nn.Conv2d(1, 3, 12)
        #self.conv22 = nn.Conv2d(3, 1, 1)
        #self.conv_shape2 = 1
        
        #self.fc1 = nn.Linear(self.conv_shape1 * self.conv_shape1 + self.conv_shape2 * self.conv_shape2 + 21, 48)
        self.fc1 = nn.Linear(self.conv_shape1 * self.conv_shape1 + 21, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 10)
        
    def forward(self, x1, x2):
        x11 = F.relu(self.conv11(x1))
        x11 = F.relu(self.conv12(x11))
        #x12 = F.relu(self.conv21(x1))
        #x12 = F.relu(self.conv22(x12))
        #import ipdb; ipdb.set_trace()
        x11 = x11.view(-1, self.conv_shape1 * self.conv_shape1)
        #x12 = x12.view(-1, self.conv_shape2 * self.conv_shape2)
        x = torch.cat((x11, x2), 1)
        #x = torch.cat((torch.cat((x11, x12), 1), x2), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x



# Test environments

test_env = pickle.load(open("maps__test.pk", "rb"))
train_env = pickle.load(open("maps__train.pk", "rb"))
rule_base_access = True


# Our method (using demo)

system1 = System1Adapted()
if rule_base_access:
	system2 = System2()
else:
	pass



#demos_water = pickle.load(open("demos_water_gold.pk", "rb"))
#demo = demos_water['1layer'][1]

#demo_model = [ fullstate(s) for s in demo ]
#for state in demo_model:
#	system1.next_state(state)
#segmentation_index, skill_sequence = system1.result()
# We're not inferring the objective ourselves, so no point


# Prepare dataset

if os.path.exists("action_clone_dataset.pk"):
	x1, x2, y = pickle.load(open("action_clone_dataset.pk", "rb"))

else:
	x1 = np.zeros((0,1,12,12))
	x2 = np.zeros((0,21))
	y = []


	demo_type_strings = ["1layer", "2layer", "3layer", "gem_gold", "grass_gold", "iron_gold", "stone_gold", "water_gold", "wood_gold"]
	for demo_string in demo_type_strings:
		demos_rule_dict = pickle.load(open("demos_" + demo_string + ".pk", "rb"))
		for i, demo in enumerate(demos_rule_dict['1layer']):

			prev_state = demo[0]
			prev_state = system1.observation_function(fullstate(prev_state))
			inventory = np.zeros(21)
			
			for state in demo[1:]:
				state = system1.observation_function(fullstate(state))
				x1 = np.append(x1, np.expand_dims(np.expand_dims(state, 0), 0), axis=0)
				x2 = np.append(x2, np.expand_dims(inventory.copy(), 0), axis=0)
				
				px, py = np.where(prev_state == 1)
				cx, cy = np.where(state == 1)
				if cy - py == 1:
					assert px == cx
					y.append(1)
				elif cy - py == -1:
					assert px == cx
					y.append(0)
				else:
					if cx - px == 1:
						assert cy == py
						y.append(3)
					elif cx - px == -1:
						assert cy == py
						y.append(2)
					elif cy == py and cx == px:
						pdx, pdy = np.where(prev_state % 1 == 0.5)
						cdx, cdy = np.where(state % 1 == 0.5)
						if pdx == cdx and pdy == cdy:
							y.append(4)
							# Also update the estimated inventory
							if rule_base_access:
								object_in_front = state[cdx, cdy] + 0.5
								rule_tr = system2.rule_dict_oracle[object_in_front][0]
								rule_pre = system2.rule_dict_oracle[object_in_front][1]
								for tr, pre in zip(rule_tr, rule_pre):
									if (inventory - pre >= 0).all():
										inventory += tr[:-1]
						else:
							# Here we go again
							if cdy - cy == 1:
								assert cx == cdx
								y.append(1)
							elif cdy - cy == -1:
								assert cx == cdx
								y.append(0)
							else:
								if cdx - cx == 1:
									assert cdy == cy
									y.append(3)
								elif cdx - cx == -1:
									assert cdy == cy
									y.append(2)
				# Now that we have the direction
				prev_state = state

	pickle.dump((x1, x2, y), open("action_clone_dataset.pk", "wb"))


# Prepare model
net = Net()
net = net.float()


load_model = True
save_model = True


if load_model and os.path.exists('mytraining.pt'):
	print("Loading Model")
	checkpoint = torch.load('mytraining.pt')
	net.load_state_dict(checkpoint['state_dict'])
else:
	## L2 loss
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# Train

	for epoch in range(1):  

	    running_loss = 0.0
	    for i in range(5000): # loop over the dataset multiple times
	        
	        # zero the parameter gradients
	        optimizer.zero_grad()

	        # forward + backward + optimize
	        y_guess = net(torch.tensor(x1).type(torch.float32), torch.tensor(x2).type(torch.float32))
	        loss = criterion(y_guess, torch.tensor(y).type(torch.LongTensor))
	        loss.backward()
	        optimizer.step()

	        # print statistics
	        running_loss += loss.item()
	        if i % 20 == 19:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 20))
	            running_loss = 0.0
	            #if np.isclose(running_loss, 0.0):
	            #	break

	print('Finished Training')
	if save_model:
		torch.save({'state_dict': net.state_dict(), 'optimizer' : optimizer.state_dict()}, \
			'mytraining.pt')
		print('Model Saved')
	else:
		pass


		
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
	action_seq = []

	for _ in range(125): # Max skills
		observable_env = system1.observation_function(fullstate(state))
		action_prob = net(torch.tensor(np.expand_dims(np.expand_dims(observable_env, 0), 0)).type(torch.float32), \
			torch.tensor(np.expand_dims(state.inventory, 0)).type(torch.float32))
		_, state = state.step(action_prob.argmax().item())
		action_seq.append(action_prob.argmax().item())

		if state.inventory[10] > 0:
			success += 1
			success_cases.append(i)
		else:
			failure += 1
			failure_cases.append(i)
	
	state.render()
	state.render()
	print("\n\n\n\n\n")
	print(action_seq)
	

print("Success:{}, Failure:{}".format(success, failure))
import ipdb; ipdb.set_trace()
