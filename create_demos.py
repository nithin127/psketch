import pickle
import numpy as np
from system1 import System1, fullstate

inventory_number = {"iron": 7, "grass": 8, "wood": 9, "gold": 10, "gem": 11, "plank": 12, "stick": 13, "axe": 14, \
			"rope": 15, "bed": 16, "shears": 17, "cloth": 18, "bridge": 19, "ladder": 20}
number_inventory = {7: "iron", 8: "grass", 9: "wood", 10: "gold", 11: "gem", 12: "plank", 13: "stick", 14: "axe", \
			15: "rope", 16: "bed", 17: "shears", 18: "cloth", 19: "bridge", 20: "ladder"}
string_num_dict = { "free": 0, "workshop0": 3, "workshop1": 4, "workshop2": 5, "iron": 6, "grass": 7, "wood": 8, "water": 9, "stone": 10, "gold": 11, "gem": 12 }
num_string_dict = { 0: "free", 3: "workshop0", 4: "workshop1", 5: "workshop2", 6: "iron", 7: "grass", 8: "wood", 9: "water", 10: "stone", 11: "gold", 12: "gem" }		


maps = pickle.load(open("maps_water_gold.pk", "rb"))
s1 = System1()


demos = {"1layer": []}

i = 0

while i < len(maps):
	state = maps[i]
	demo = [state]
	save = True
	redo = False
	while(True):
		agent_obs = s1.observation_function(fullstate(state))
		x, y = np.where(agent_obs%1 == 0.5)
		pos_x, pos_y = np.where(agent_obs == 1)
		agent_obs_modified = agent_obs.copy()
		agent_obs_modified[x[0],y[0]] += 0.5
		state.render()
		state.render()
		options_x, options_y = np.where(agent_obs_modified >= 3)
		print("")
		option_num = input("Select option: \n{}\n\n Else, enter \"k\" to move on, \"s\" to save \"r\" to redo\n".\
			format([(i, num_string_dict[agent_obs_modified[x,y]], (x,y)) for i, (x,y) in enumerate(zip(options_x, options_y))]))
		if option_num == "s":
			break
		elif option_num == "k":
			save = False
			break
		elif option_num == "r":
			save = False
			redo = True
			break
		else:
			option_num = int(option_num)
			action_seq = s1.use_object(agent_obs, (pos_x[0], pos_y[0]), (options_x[option_num], options_y[option_num]))
			for act in action_seq:
				_, state = state.step(act)
				demo.append(state)
	if save:
		demos["1layer"].append(demo)
	if not redo:
		i+=1


pickle.dump(demos, open("demos_water_gold.pk", "wb"))
