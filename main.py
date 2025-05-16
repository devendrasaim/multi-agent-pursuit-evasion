import numpy as np 
import pandas as pd
import random
import os

from planners.tom import PlannerAgent as Tom
from planners.jerry import PlannerAgent as Jerry
from planners.planner import PlannerAgent as Spike

def manhattan_distance(p1, p2):
    """Computes the Manhattan distance between two points (row, col)."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def select_valid_locations(grid, num_points=3, min_distance=5):
    # Find all valid (non-obstacle) positions
    free_positions = list(map(tuple, np.argwhere(grid == 0)))
    if len(free_positions) < num_points:
        raise ValueError("Not enough free spaces to select locations!")

    selected_positions = []
    first_pos = random.choice(free_positions)
    selected_positions.append(first_pos)
    while len(selected_positions) < num_points:
        candidate = random.choice(free_positions)
        # Ensure the candidate is at least `min_distance` away from all selected positions
        if all(manhattan_distance(candidate, pos) >= min_distance for pos in selected_positions):
            selected_positions.append(candidate)

    return np.array(selected_positions)

class Task:
    """Base class of a task"""
    def __init__(self, id, running_id,
                 max_iter=1000, prob=[0.3,0.3,0.4]):
        self.id = id
        self.running_id = running_id
        self.max_iter = max_iter
        self.prob = prob

        self.world = np.load("data/grid_files/grid_"+str(id)+".npy")
        self.obstacles = np.array(np.where(self.world==1)).T

        self.agents = [Tom(), Jerry(), Spike()]

    def save_log(self):
        if not os.path.exists("data/proj_iii_solutions/"):
            os.makedirs("data/proj_iii_solutions/")
        data = np.array([np.hstack(self.state_log[i]) for i in range(len(self.state_log))])
        path_df = pd.DataFrame(data=data, columns=["Tom_X", "Tom_Y", "Jerry_X", "Jerry_Y", "Spike_X", "Spike_Y"])
        path_df.to_csv("data/proj_iii_solutions/"+str(self.id)+"_"+str(self.running_id)+".csv", index=False)

    def reset(self):
        self.state = select_valid_locations(self.world, num_points=3, min_distance=3)
        self.legal_agents = [True, True, True]
        self.state_log = [self.state]

    def mod_action(self, a):
        mod_action_idx = np.random.choice(3,p=self.prob)
        if mod_action_idx == 1:
            return a 
        if mod_action_idx == 0:
            return np.array([-a[1], a[0]])
        if mod_action_idx == 2:
            return np.array([a[1], -a[0]])

    def step(self):
        tom_a = self.mod_action(self.agents[0].plan_action(self.world, self.state[0], self.state[1], self.state[2]))
        jerry_a = self.mod_action(self.agents[1].plan_action(self.world, self.state[1], self.state[2], self.state[0]))
        spike_a = self.mod_action(self.agents[2].plan_action(self.world, self.state[2], self.state[0], self.state[1]))
        
        actions = [tom_a, jerry_a, spike_a]

        is_a_valid = np.array([self.check_action(self.state[0], tom_a),
                               self.check_action(self.state[1], jerry_a),
                               self.check_action(self.state[2], spike_a)])
        
        for a in range(3):
            if is_a_valid[a]:
                self.state[a] += actions[a]

    def run(self):
        self.reset()
        done = False
        tom_wins, jer_wins, spk_wins = False, False, False
        while not done:
            self.step()
            self.state_log += [np.copy(self.state)]

            collisions = [self.check_collision(self.state[0]),
                          self.check_collision(self.state[1]),
                          self.check_collision(self.state[2])]

            tom_wins = np.all(self.state[0]==self.state[1]) and not collisions[0]
            jer_wins = np.all(self.state[1]==self.state[2]) and not collisions[1]
            spk_wins = np.all(self.state[2]==self.state[0]) and not collisions[2]

            if np.any(np.array([tom_wins, jer_wins, spk_wins])):
                done = True
            if np.any(np.array(collisions)):
                done = True
            if len(self.state_log)==self.max_iter:
                done = True 

        self.save_log()
        if tom_wins and (not jer_wins) and (not spk_wins):
            return np.array([3, 0, 0])
        if jer_wins and (not tom_wins) and (not spk_wins):
            return np.array([0, 3, 0])
        if spk_wins and (not tom_wins) and (not jer_wins):
            return np.array([0, 0, 3])
        if np.any(np.array(collisions)):
            return np.array([1-1*collisions[0], 1-1*collisions[1], 1-1*collisions[2]])

        return np.array([1, 1, 1])

    def check_action(self, state, action):
        state_ = state + action
        is_valid_action = np.all(abs(action)<=1)
        is_inbound = np.min(state_)>=0 and np.max(state_)<=29

        return is_valid_action and is_inbound
    
    def check_collision(self, state):
        return np.any(np.all(self.obstacles==state, axis=1))

if __name__ == "__main__":
    # Test all grid tasks with each task tested for 5 times
    for id in range(100):
        for running_id in range(5):
            T = Task(id, running_id)
            result = T.run()
            print (id, running_id, result)