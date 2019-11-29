import numpy as np
from craft.envs.craft_world import *
from craft.envs.cookbook import Cookbook


WIDTH = 15
HEIGHT = 15

cookbook = Cookbook()
random = np.random.RandomState(0)
world = CraftWorld()


# Design environment here

def design_env():
	# Assuming we have access to all objects
	grid = np.zeros((WIDTH, HEIGHT, cookbook.n_kinds))
	i_bd = cookbook.index["boundary"]


	grid[0, :, i_bd] = 1
	grid[WIDTH-1:, :, i_bd] = 1
	grid[:, 0, i_bd] = 1
	grid[:, HEIGHT-1:, i_bd] = 1


	for primitive in cookbook.primitives:
		for _ in range(3):
			ws_x, ws_y = random_free(grid, random)
			grid[ws_x, ws_y, primitive] = 1


	for environment_obj in cookbook.environment:
		ws_x, ws_y = random_free(grid, random)
		grid[ws_x, ws_y, environment_obj] = 1


	init_pos = random_free(grid, random)

	return CraftScenario(grid, init_pos, world)


def save_state(state, name="../../wood_one.pk"):
	import pickle
	pickle.dump(state, open(name, "wb"))

# Create the environment and save

scenario = design_env()
state = scenario.init()
state.render()

import ipdb; ipdb.set_trace()
state.render()
save_state(state, "custom_map.pk")
