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
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 1, 3)
        self.fc1 = nn.Linear(8 * 8 + 21, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x1, x2):
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(-1, 8 * 8)
        x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x



# Single demonstration

demos_water = pickle.load(open("demos_water_gold.pk", "rb"))
demo = demos_water['1layer'][1]


# Test environments

test_env = pickle.load(open("maps__test.pk", "rb"))
train_env = pickle.load(open("maps__train.pk", "rb"))
rule_base_access = False


# Our method (using demo)

system1 = System1Adapted()
if rule_base_access:
	system2 = System2()
else:
	pass


#demo_model = [ fullstate(s) for s in demo ]
#for state in demo_model:
#	system1.next_state(state)
#segmentation_index, skill_sequence = system1.result()
# We're not inferring the objective ourselves, so no point


# Prepare dataset

x1 = np.zeros((0,1,12,12))
x2 = np.zeros((0,21))
y = []


demo_type_strings = ["1layer", "2layer", "3layer", "gem_gold", "grass_gold", "iron_gold", "stone_gold", "water_gold", "wood_gold"]
for demo_string in [demo_type_strings[0]]:
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
		

# Prepare model
net = Net()
net = net.float()

## L2 loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(1000):
        
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

print('Finished Training')





		
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
