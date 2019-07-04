#### Objects dictionary:
#0 free
#1 boundary
#2 workshop0
#3 workshop1
#4 workshop2
#5 water
#6 stone
#### These items are primitive, can be found in the grid
#7 iron
#8 grass
#9 wood
#10 gold
#11 gem
#### These items are made in workshops
#12 plank
#13 stick
#13 axe
#15 rope
#16 bed
#17 shears
#18 cloth
#19 bridge
#20 ladder

import numpy as np

NOISE_CHOICE = 0.3
NOISE_PATH = 0.3

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4

WIDTH = 12
HEIGHT = 12

## Helper functions ##
def neighbors(pos, dirc=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dirc is None or dirc == LEFT):
        neighbors.append((x-1, y, LEFT))
    if y > 0 and (dirc is None or dirc == DOWN):
        neighbors.append((x, y-1, DOWN))
    if x < WIDTH - 1 and (dirc is None or dirc == RIGHT):
        neighbors.append((x+1, y, RIGHT))
    if y < HEIGHT - 1 and (dirc is None or dirc == UP):
        neighbors.append((x, y+1, UP))
    return neighbors

def get_prev(pos, dirc):
	if dirc == 0:
		return (pos[0], pos[1] + 1)
	elif dirc == 1:
		return (pos[0], pos[1] - 1)
	elif dirc == 2:
		return (pos[0] + 1, pos[1])
	elif dirc == 3:
		return (pos[0] - 1, pos[1])

def get_map(state, goal):
	obstacle_map = np.zeros(state.grid.shape[:-1])
	for i in range(obstacle_map.shape[0]):
		for j in range(obstacle_map.shape[1]):
			if np.any(state.grid[i][j]):
				obstacle_map[i][j] = 1
	obstacle_map[goal[0], goal[1]] = 5
	return obstacle_map

def planner(start, goal, obs_map):
	"""
	Dijstra's implementation
	"""
	cost_map = np.inf*np.ones(obs_map.shape)
	dir_map = np.zeros(obs_map.shape)
	cost_map[start[0],start[1]] = 0
	to_visit = []
	to_visit.append(start)
	while len(to_visit) > 0:
		curr = to_visit.pop(0)
		for nx, ny, d in neighbors(curr, None):
			if obs_map[nx, ny] == 1:
				continue
			cost = cost_map[curr[0],curr[1]] + 1
			if cost < cost_map[nx,ny]:
				if obs_map[nx, ny] == 0:
					to_visit.append((nx, ny))
				dir_map[nx,ny] = d
				cost_map[nx,ny] = cost
	seq = [4]
	curr = goal
	d = dir_map[curr[0],curr[1]]
	curr = get_prev(curr, d)
	if not dir_map[curr[0],curr[1]] == d:
		seq.append(d)
	while not curr == start:
		d = dir_map[curr[0],curr[1]]
		curr = get_prev(curr, d)
		seq.append(d)
	seq.reverse()
	return seq	

def get_goal(state, obj):
	xs, ys = np.where(state.grid[:,:,obj] == 1)
	xi, yi = state.pos
	if not xs.any():
		#print("No goal available")
		return None
	# Choose one; either the best or a suboptimal location
	#	 Probably a bad way to choose stuff
	t = np.argmin((xs-xi)**2 + (ys-yi)**2)
	if np.random.random() < NOISE_CHOICE:
		t = np.random.choice(len(xs))
	return (xs[t], ys[t])

## Subpolicies ##

def get_wood(state):
	goal = get_goal(state, 9)
	obstacle_map = get_map(state, goal)
	return planner(state.pos, goal, obstacle_map)

def get_iron(state):
	goal = get_goal(state, 7)
	obstacle_map = get_map(state, goal)
	return planner(state.pos, goal, obstacle_map)

def get_grass(state):
	goal = get_goal(state, 8)
	obstacle_map = get_map(state, goal)
	return planner(state.pos, goal, obstacle_map)

def make0(state):
	goal = get_goal(state, 2)
	obstacle_map = get_map(state, goal)
	return planner(state.pos, goal, obstacle_map)

def make1(state):
	goal = get_goal(state, 3)
	obstacle_map = get_map(state, goal)
	return planner(state.pos, goal, obstacle_map)

def make2(state):
	goal = get_goal(state, 4)
	obstacle_map = get_map(state, goal)
	return planner(state.pos, goal, obstacle_map)

def get_gold(state):
	goal = get_goal(state, 10)
	if not goal:
		return None
	new_goal = []
	goal_scores = []
	for nx, ny, d in neighbors(goal):
		goal_score = 1
		if not state.grid[nx,ny].argmax() == 1:
			new_goal.append((nx,ny))
			for n2x, n2y, _ in neighbors(goal):
				if state.grid[n2x,n2y].argmax() == 0:
					goal_score += 3
			goal_scores.append(goal_score)
	new_goal = new_goal[np.random.choice(len(new_goal), p=np.array(goal_scores)/sum(goal_scores))]
	# Change logic: pick the closest one that is not blocked
	obstacle_map = get_map(state, new_goal)
	seq = planner(state.pos, new_goal, obstacle_map)
	if state.inventory[19] > 0:
		seq.append(seq[-2])
		seq.append(4)
	else:
		#print("Insufficient inventory")
		del seq[-1]
	return seq

def get_gem(state):
	goal = get_goal(state, 11)
	if not goal:
		return None
	new_goal = []
	goal_scores = []
	for nx, ny, d in neighbors(goal):
		goal_score = 1
		if not state.grid[nx,ny].argmax() == 1:
			new_goal.append((nx,ny))
			for n2x, n2y, _ in neighbors(goal):
				if state.grid[n2x,n2y].argmax() == 0:
					goal_score += 3
			goal_scores.append(goal_score)
	new_goal = new_goal[np.random.choice(len(new_goal), p=np.array(goal_scores)/sum(goal_scores))]
	obstacle_map = get_map(state, new_goal)
	seq = planner(state.pos, new_goal, obstacle_map)
	if state.inventory[14] > 0:
		seq.append(seq[-2])
		seq.append(4)
	else:
		#print("Insufficient inventory")
		del seq[-1]
	return seq


