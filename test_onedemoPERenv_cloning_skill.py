import os, time
import pickle
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

if os.path.exists("skill_clone_dataset.pk"):
	x1, x2, y = pickle.load(open("skill_clone_dataset.pk", "rb"))

else:
	x1 = np.zeros((0,1,12,12))
	x2 = np.zeros((0,21))
	y = []


	demo_type_strings = ["1layer", "2layer", "3layer", "gem_gold", "grass_gold", "iron_gold", "stone_gold", "water_gold", "wood_gold"]
	for demo_string in demo_type_strings:
		demos_rule_dict = pickle.load(open("demos_" + demo_string + ".pk", "rb"))
		for i, demo in enumerate(demos_rule_dict['1layer']):
			demo_model = [ fullstate(s) for s in demo ]
			for state in demo_model:
				system1.next_state(state)
			segmentation_index, skill_sequence = system1.result()
			system1.restart()
			segmentation_index = [0] + segmentation_index
			inventory = np.zeros(21)
			for index, skill in zip(segmentation_index, skill_sequence):
				curr_state = system1.observation_function(demo_model[index])
				x1 = np.append(x1, np.expand_dims(np.expand_dims(curr_state, 0), 0), axis=0)
				x2 = np.append(x2, np.expand_dims(inventory.copy(), 0), axis=0)
				# Minus 3 is to adjust the classification categories
				y.append(skill['object_before'] - 3)
				# Update inventory as per rule base
				if rule_base_access:
					rule_tr = system2.rule_dict_oracle[skill['object_before']][0]
					rule_pre = system2.rule_dict_oracle[skill['object_before']][1]
					for tr, pre in zip(rule_tr, rule_pre):
						if (inventory - pre >= 0).all():
							inventory += tr[:-1]
				else:
					pass
	
	pickle.dump((x1, x2, y), open("skill_clone_dataset.pk", "wb"))


# Prepare model
net = Net()
net = net.float()


load_model = False
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

	        ##
	        # Jumble dataset !
	        ##

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
			'mytraining_2.pt')
		print('Model Saved')
	else:
		pass


		
success = 0
success_cases = []
failure = 0
failure_cases = []
total_time = 0


for i, env in enumerate(train_env):
#for i, env in enumerate(test_env):
	start = time.time()
	state = env
	observable_env = system1.observation_function(fullstate(state))
	state.render()
	state.render()
	import ipdb; ipdb.set_trace()
	input("\n\n\n\nEnvironment number: {}\n\n\n\n\n".format(i))
	skill_seq = []
	sequence_length = 0
	for _ in range(25): # Max skills
		observable_env = system1.observation_function(fullstate(state))
		skill_prob = net(torch.tensor(np.expand_dims(np.expand_dims(observable_env, 0), 0)).type(torch.float32), \
			torch.tensor(np.expand_dims(state.inventory, 0)).type(torch.float32))

		done = False
		possible_objects = np.where(observable_env == skill_prob.argmax().item() + 3)
		for skill_param_x, skill_param_y in zip(possible_objects[0], possible_objects[1]):
			pos_x, pos_y = np.where(observable_env == 1)
			if done:
				break
			try:
				action_seq = system1.use_object(observable_env, (pos_x[0], pos_y[0]), (skill_param_x, skill_param_y))
				if len(action_seq) > 0 and action_seq[-1] == 4:
					done = True
					for a in action_seq:
						_, state = state.step(a)
						sequence_length += 1
					skill_seq.append(skill_prob.argmax().item() + 3)
					break
			except:
				print("Skill_params: {} failed".format((skill_param_x, skill_param_y)))
				pass
				
		if state.inventory[10] > 0:
			success += 1
			success_cases.append((i, sequence_length))
			total_time += end - start
			break

	if state.inventory[10] == 0:
		failure += 1
		failure_cases.append(i)
	
	state.render()
	state.render()
	print("\n\n\n\n\n")
	print(skill_seq)
	

print("\n\n\n\n")
for s in success_cases: print(s)
if success > 0:
	print("Avg. time taken: {}, Success:{}, Failure:{}".format(total_time/success, success, failure))
else:
	print("Success:{}, Failure:{}".format(success, failure))
import ipdb; ipdb.set_trace()
