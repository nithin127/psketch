import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from system1 import *



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm2d(1)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        #def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #    return (size - (kernel_size - 1) - 1) // stride  + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = 4 #convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        #import ipdb; ipdb.set_trace()
        return self.head(x.view(x.size(0), -1))



BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
screen_height = 10
screen_width = 10

# Get number of actions from gym action space
n_actions = 10

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

demo_type_strings = ["1layer", "2layer", "3layer", "gem_gold", "grass_gold", "iron_gold", "stone_gold", "water_gold", "wood_gold"]


demos = {}
for demo_string in demo_type_strings:
    demos[demo_string] = pickle.load(open("demos_" + demo_string + ".pk", "rb"))

system1 = System1()


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    print("Episode: ", i_episode)
    actual_state = np.random.choice(demos[np.random.choice(demo_type_strings)]['1layer'])[0]
    
    current_screen = system1.observation_function(fullstate(actual_state))
    current_screen = torch.tensor(np.expand_dims(np.expand_dims(current_screen, 0), 0)).type(torch.float32)
    last_screen = system1.observation_function(fullstate(actual_state))
    last_screen = torch.tensor(np.expand_dims(np.expand_dims(last_screen, 0), 0)).type(torch.float32)

    state = current_screen - last_screen


    for t in count():
        # Select and perform an action
        action = select_action(state)

        done_skill = False
        skill_seq = []
        possible_objects = np.where(current_screen[0][0] == action.item() + 3)
        for skill_param_x, skill_param_y in zip(possible_objects[0], possible_objects[1]):
            pos_x, pos_y = np.where(current_screen[0][0] == 1)
            if done_skill:
                break
            try:
                action_seq = system1.use_object(current_screen, (pos_x[0], pos_y[0]), (skill_param_x, skill_param_y))
                if len(action_seq) > 0 and action_seq[-1] == 4:
                    done_skill = True
                    print(action_seq)
                    for a in action_seq:
                        _, actual_state = actual_state.step(a)
                    current_screen = system1.observation_function(fullstate(actual_state))
                    skill_seq.append(skill_prob.argmax().item() + 3)
                    break
            except:
                #print("Skill_params: {} failed".format((skill_param_x, skill_param_y)))
                pass


        reward = 1 if actual_state.inventory[10] > 0 else 0
        if reward == 1:
            done = True
        else:
            done = False

        # Observe new state
        last_screen = current_screen
        current_screen = system1.observation_function(fullstate(actual_state))
        current_screen = torch.tensor(np.expand_dims(np.expand_dims(current_screen, 0), 0)).type(torch.float32)

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, torch.tensor([reward]).type(torch.float32))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done or t > 3000:
            episode_durations.append(t + 1)
            #plot_durations()
            break
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
actual_state.render()

