import os
import gym
import yaml
import curses
import numpy as np
from misc import util
from misc.experience import Transition

from collections import defaultdict, namedtuple
import itertools
from craft.envs.craft_world import CraftWorld
from craft.envs.cookbook import Cookbook

Task = namedtuple("Task", ["goal", "steps"])

# Shift these things to a config file?; 
# Note that craft_world.py also uses the same set of parameters
WIDTH = 12
HEIGHT = 12

WINDOW_WIDTH = 5
WINDOW_HEIGHT = 5

N_WORKSHOPS = 3

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4
N_ACTIONS = USE + 1


class CraftEnv(gym.Env):
    metadata = {'render.modes': ['human']} # What does this mean?

    def __init__(self):
        self.world = CraftWorld()
        self.cookbook = Cookbook()
        self.subtask_index = util.Index()
        self.task_index = util.Index()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "hints.yaml")) as hints_f:
            self.hints = yaml.load(hints_f)

        # initialize randomness
        # self.random = np.random.RandomState(0) 
        # Think about this

        # organize task and subtask indices
        self.tasks_by_subtask = defaultdict(list)
        self.tasks = []
        for hint_key, hint in self.hints.items():
            goal = util.parse_fexp(hint_key)
            goal = (self.subtask_index.index(goal[0]), self.cookbook.index[goal[1]])
            steps = [self.subtask_index.index(a) for a in hint]
            steps = tuple(steps)
            task = Task(goal, steps)
            for subtask in steps:
                self.tasks_by_subtask[subtask].append(task)
            self.tasks.append(task)
            self.task_index.index(task)

    def reset(self, task = None, difficulty = 3):
        """
        Either give the task; or give the difficulty level == 1,2, or 3
        """
        if not task:
            # These difficulty levels are hard-coded of sorts, can easily write logic to overcome this
            if difficulty == 1:
                task = self.tasks[np.random.choice(4)]
            elif difficulty == 2:
                task = self.tasks[4 + np.random.choice(4)]
            elif difficulty == 3:
                task = self.tasks[8 + np.random.choice(2)]
            else:
                task = self.tasks[np.random.choice(len(self.tasks))]
        self.goal, _ = task
        goal_name, goal_arg = self.goal
        scenario = self.world.sample_scenario_with_goal(goal_arg)
        self.state = scenario.init()
        
        return self.state#, task

    def step(self, action):
        r, s = self.state.step(action)
        pr_s = self.state
        self.state = s
        inv = np.where(s.inventory - pr_s.inventory)[0]
        if inv:
            print("Gathered {}".format(self.cookbook.index.ordered_contents[int(inv)-1]))
        return s, r, s.satisfies(self.goal), {}
        
    def render(self, mode="human", close=False):
        # Different modes -- Render in terminal vs a separate window
        def _visualize(win):
            curses.start_color()
            for i in range(1, 8):
                curses.init_pair(i, i, curses.COLOR_BLACK)
                curses.init_pair(i+10, curses.COLOR_BLACK, i)
            state = self.state
            win.clear()
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if not (state.grid[x, y, :].any() or (x, y) == state.pos):
                        continue
                    thing = state.grid[x, y, :].argmax()
                    if (x, y) == state.pos:
                        if state.dir == LEFT:
                            ch1 = "<"
                            ch2 = "@"
                        elif state.dir == RIGHT:
                            ch1 = "@"
                            ch2 = ">"
                        elif state.dir == UP:
                            ch1 = "^"
                            ch2 = "@"
                        elif state.dir == DOWN:
                            ch1 = "@"
                            ch2 = "v"
                        color = curses.color_pair(0)
                    elif thing == self.cookbook.index["boundary"]:
                        ch1 = ch2 = curses.ACS_BOARD
                        color = curses.color_pair(10 + thing)
                    else:
                        name = self.cookbook.index.get(thing)
                        ch1 = name[0]
                        ch2 = name[-1]
                        color = curses.color_pair(10 + thing)

                    win.addch(HEIGHT-y, x*2, ch1, color)
                    win.addch(HEIGHT-y, x*2+1, ch2, color)
            win.refresh()
            
        curses.wrapper(_visualize)
        inventory = self.state.inventory
        
        print("Inventory:\nIron:{}   Grass:{}   Wood:{}   Gold:{}   Gem:{}   Plank:{}   Stick:{}   Axe:{}   Rope:{}   Bed:{}   Shears:{}   Cloth:{}   Bridge:{}   Ladder:{}"\
            .format(inventory[7], inventory[8], inventory[9], inventory[10], inventory[11], inventory[12], inventory[13], inventory[14], inventory[15],\
                inventory[16], inventory[17], inventory[18], inventory[19], inventory[20]))