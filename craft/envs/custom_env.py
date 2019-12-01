import numpy as np
from craft.envs.craft_world import *
from craft.envs.cookbook import Cookbook


WIDTH = 15
HEIGHT = 15

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

	for primitive in cookbook.primitives:
		for _ in range(num_prim):
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
custom_maps = []
for _ in range(10):
	custom_map = design_env(3)
	custom_maps.append(custom_map)

save_state(custom_maps, "custom_maps.pk")
import ipdb; ipdb.set_trace()

state = scenario.init()
state.render()
state.render()
save_state(state, "custom_map.pk")
