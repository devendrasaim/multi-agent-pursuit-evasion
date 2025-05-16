import numpy as np
from typing import List, Tuple, Optional
import math
from functools import lru_cache

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0
        self.action = None
        self._cached_actions = None

    @property
    def actions(self) -> List[Tuple[int,int]]:
        if self._cached_actions is None:
            self._cached_actions = [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        return self._cached_actions

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.actions)

    def expand(self) -> 'Node':
        tried = {child.action for child in self.children}
        for a in self.actions:
            if a not in tried:
                next_state = self.sim(self.state, a)
                child = Node(next_state, parent=self)
                child.action = a
                self.children.append(child)
                return child
        raise RuntimeError("No actions left to expand")

    def best_child(self, c_param=1.4) -> 'Node':
        if not self.children:
            return None
        choices = []
        for child in self.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = (child.value/child.visits) + c_param * math.sqrt(2*math.log(self.visits)/child.visits)
            choices.append(uct)
        return self.children[int(np.argmax(choices))]

    @staticmethod
    @lru_cache(maxsize=1024)
    def apply_probabilistic_transition(action, prob_idx):
        if prob_idx == 1:
            return action
        elif prob_idx == 0:
            return (-action[1], action[0])
        else:
            return (action[1], -action[0])

    def rollout_policy(self, actions, prob=[0.3, 0.3, 0.4]):
        grid, cur, pursued, pursuer = self.state
        
        next_positions = [(cur[0] + a[0], cur[1] + a[1]) for a in actions]
        target_distances = [(np[0] - pursued[0])**2 + (np[1] - pursued[1])**2 for np in next_positions]
        pursuer_distances = [(np[0] - pursuer[0])**2 + (np[1] - pursuer[1])**2 for np in next_positions]
        
        scores = []
        for i, (td, pd) in enumerate(zip(target_distances, pursuer_distances)):
            if td < 2.25:
                return self.apply_probabilistic_transition(actions[i], np.random.choice(3, p=prob))
            if pd < 2.25:
                continue
            score = -td + pd
            scores.append(score)
        
        if not scores:
            return self.apply_probabilistic_transition(actions[0], np.random.choice(3, p=prob))
        
        scores = np.array(scores)
        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()

        chosen_action = actions[np.random.choice(len(actions), p=probs)]
        return self.apply_probabilistic_transition(chosen_action, np.random.choice(3, p=prob))

    def sim(self, state, action, prob=[0.3, 0.3, 0.4]):
        grid, cur, pursued, pursuer = state
        rows, cols = grid.shape
        
        mod_action = self.apply_probabilistic_transition(action, np.random.choice(3, p=prob))
        nxt_cur = (cur[0] + mod_action[0], cur[1] + mod_action[1])
        
        if not (0 <= nxt_cur[0] < rows and 0 <= nxt_cur[1] < cols) or grid[nxt_cur] == 1:
            nxt_cur = cur

        tom_action = self.move(grid, pursued, pursuer)
        tom_mod_action = self.apply_probabilistic_transition(tom_action, np.random.choice(3, p=prob))
        nxt_pursued = (pursued[0] + tom_mod_action[0], pursued[1] + tom_mod_action[1])
        if not (0 <= nxt_pursued[0] < rows and 0 <= nxt_pursued[1] < cols) or grid[nxt_pursued] == 1:
            nxt_pursued = pursued
        
        jerry_action = self.move(grid, pursuer, cur)
        jerry_mod_action = self.apply_probabilistic_transition(jerry_action, np.random.choice(3, p=prob))
        nxt_pursuer = (pursuer[0] + jerry_mod_action[0], pursuer[1] + jerry_mod_action[1])
        if not (0 <= nxt_pursuer[0] < rows and 0 <= nxt_pursuer[1] < cols) or grid[nxt_pursuer] == 1:
            nxt_pursuer = pursuer
        
        return (grid, nxt_cur, nxt_pursued, nxt_pursuer)

    @staticmethod
    @lru_cache(maxsize=1024)
    def move(grid, pos, target):
        rows, cols = grid.shape
        all_actions = [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)][1:]
        best_action = (0, 0)
        best_dist = float('inf')
        
        for a in all_actions:
            np_ = (pos[0] + a[0], pos[1] + a[1])
            if 0 <= np_[0] < rows and 0 <= np_[1] < cols and grid[np_] == 0:
                dist = (np_[0] - target[0])**2 + (np_[1] - target[1])**2
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
        
        return best_action

    def rollout(self, prob=[0.3, 0.3, 0.4]) -> float:
        state = self.state
        grid, cur, pursued, pursuer = state
        depth = 15
        
        for _ in range(depth):
            if cur == pursued and grid[cur] == 0:
                return 1.0
            if cur == pursuer or grid[cur] == 1:
                return -1.0
            
            target_dist = (cur[0] - pursued[0])**2 + (cur[1] - pursued[1])**2
            pursuer_dist = (cur[0] - pursuer[0])**2 + (cur[1] - pursuer[1])**2
            
            if target_dist < 4:
                return 0.8
            if pursuer_dist < 4:
                return -0.8
            
            a = self.rollout_policy(self.actions, prob)
            state = self.sim(state, a, prob)
            grid, cur, pursued, pursuer = state
            
        return 0.0

    def bp(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.bp(reward)


class MCTS:
    def __init__(self, iterations=150, prob=[0.3, 0.3, 0.4]):
        self.iterations = iterations
        self.prob = prob

    def search(self, init_state):
        root = Node(init_state)
        for _ in range(self.iterations):
            node = root
            while node.children and node.is_fully_expanded():
                node = node.best_child()
            if node.visits > 0 and not node.is_fully_expanded():
                node = node.expand()
            reward = node.rollout(self.prob)
            node.bp(reward)
        best = root.best_child(c_param=0.0)
        return best.action if best else (0, 0)


class PlannerAgent:
    def __init__(self):
        self.prob = [0.3, 0.3, 0.4] 
        self._cached_directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                          [-1, -1], [-1, 1], [1, -1], [1, 1]])
    
    def plan_action(self, world: np.ndarray, current: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes an action to take from the current position to capture the pursued while evading from the pursuer.
        Handles probabilistic transitions according to the readme.

        Parameters:
        - world (np.ndarray): A 2D numpy array representing the grid environment.
        - 0 represents a walkable cell.
        - 1 represents an obstacle.
        - current (np.ndarray): The (row, column) coordinates of the current position.
        - pursued (np.ndarray): The (row, column) coordinates of the agent to be pursued.
        - pursuer (np.ndarray): The (row, column) coordinates of the agent to evade from.

        Returns:
        - np.ndarray: one of the 9 actions from 
                              [0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                [-1, -1], [-1, 1], [1, -1], [1, 1]
        """
        
        current_pos = (int(current[0]), int(current[1]))
        pursued_pos = (int(pursued[0]), int(pursued[1]))
        pursuer_pos = (int(pursuer[0]), int(pursuer[1]))
        
        mcts = MCTS(iterations=150, prob=self.prob)
        init_state = (world, current_pos, pursued_pos, pursuer_pos)
        
        try:
            best_action = mcts.search(init_state)
            return np.array(best_action)
        except Exception as e:
            return self._cached_directions[np.random.choice(9)]


