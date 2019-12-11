import numpy as np
from craft.envs.craft_world import *
from craft.envs.cookbook import Cookbook


WIDTH = 12
HEIGHT = 12

cookbook = Cookbook()
random = np.random.RandomState(0)
world = CraftWorld()


# Design environment here

def design_env(num_prim = 3):
	# Assuming we have access to all objects
	grid = np.zeros((WIDTH, HEIGHT, cookbook.n_kinds))
	i_bd = cookbook.index["boundary"]

	grid[0, :, i_bd] = 1
	grid[WIDTH-1:, :, i_bd] = 1
	grid[:, 0, i_bd] = 1
	grid[:, HEIGHT-1:, i_bd] = 1

	'''for primitive in cookbook.primitives:
		for _ in range(num_prim):
			ws_x, ws_y = random_free(grid, random)
			grid[ws_x, ws_y, primitive] = 1

	for environment_obj in cookbook.environment:
		ws_x, ws_y = random_free(grid, random)
		grid[ws_x, ws_y, environment_obj] = 1'''

	grid[1,1,11] = 1 # Gem
	grid[1,2,9] = 1 # Wood
	grid[2,1,8] = 1 # Grass
	grid[3,1,8] = 1 # Grass

	# Iron
	ws_x, ws_y = random_free(grid, random)
	grid[ws_x, ws_y, 7] = 1
	# w1
	ws_x, ws_y = random_free(grid, random)
	grid[ws_x, ws_y, 3] = 1
	# w0
	ws_x, ws_y = random_free(grid, random)
	grid[ws_x, ws_y, 2] = 1
	
	init_pos = random_free(grid, random)
	return CraftScenario(grid, init_pos, world)


def save_state(state, name="../../wood_one.pk"):
	import pickle
	pickle.dump(state, open(name, "wb"))

custom_map = design_env()
save_state(custom_map.init(), "env4.pk")
import ipdb; ipdb.set_trace()

state = custom_map.init()
state.render()
state.render()
