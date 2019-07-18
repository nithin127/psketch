import pickle
import numpy as np
from agent_level_4 import Agent_Level_4
from create_dataset import fullstate

state = pickle.load(open("wood_three.pk", "rb"))
agent = Agent_Level_4()

s0 = fullstate(state)
s0_observe = agent.observation_function(s0)

wdx, wdy = np.where(s0_observe == 8)

seq = agent.use_object(s0_observe, np.where(s0_observe == 1), (wdx[0], wdy[0]))

demo = []
demo.append(state)

for action in seq:
	_, state = state.step(action)
	demo.append(state)

import ipdb; ipdb.set_trace()
pickle.dump(demo, open("wood_three_demo.pk", "wb"))