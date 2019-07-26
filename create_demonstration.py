import pickle
import numpy as np
from main import Agent, fullstate
from craft.envs.craft_world import CraftScenario, CraftWorld
from craft.envs.cookbook import Cookbook


WIDTH = 12
HEIGHT = 12


def random_free(grid, random):
    pos = None
    while pos is None:
        (x, y) = (random.randint(WIDTH), random.randint(HEIGHT))
        if grid[x, y, :].any():
            continue
        pos = (x, y)
    return pos


def create_map(map_name = "iron_one", num_obstacles_random = 5):
	cw = CraftWorld()
	grid = np.zeros((WIDTH, HEIGHT, cw.cookbook.n_kinds))
	i_bd = cw.cookbook.index["boundary"]
	grid[0, :, i_bd] = 1
	grid[WIDTH-1:, :, i_bd] = 1
	grid[:, 0, i_bd] = 1
	grid[:, HEIGHT-1:, i_bd] = 1
	for _ in range(num_obstacles_random):
		ws_x, ws_y = random_free(grid, cw.random)
		grid[ws_x, ws_y, cw.cookbook.index["boundary"]] = 1

	ws_x, ws_y = random_free(grid, cw.random)
	grid[ws_x, ws_y, cw.cookbook.index["iron"]] = 1

	init_pos = random_free(grid, cw.random)
	scenario = CraftScenario(grid, init_pos, cw)
	state = scenario.init()
	state.render()
	import ipdb; ipdb.set_trace()
	state.render()
	pickle.dump(state, open(map_name + ".pk", "wb"))
	


def create_demo(map_name = "iron_one"):	
	state = pickle.load(open(map_name + ".pk", "rb"))
	agent = Agent(None)

	s0 = fullstate(state)
	s0_observe = agent.observation_function(s0)
	wdx, wdy = np.where(s0_observe == 6)
	seq = agent.use_object(s0_observe, np.where(s0_observe == 1), (wdx[0], wdy[0]))
	demo = []
	demo.append(state)

	for action in seq:
		_, state = state.step(action)
		state.render()
		demo.append(state)

	state.render()
	import ipdb; ipdb.set_trace()
	pickle.dump(demo, open(map_name + "_demo.pk", "wb"))


if __name__ == "__main__":
	#create_map()
	create_demo()