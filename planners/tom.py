import numpy as np
from typing import List, Tuple, Optional

def dfs(grid, start, end):
    """A DFS example"""
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    visited = set()
    parent = {start: None}

    # Consider all 8 possible moves (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal moves

    while stack:
        x, y = stack.pop()
        if (x, y) == end:
            # Reconstruct the path
            path = []
            while (x, y) is not None:
                path.append((x, y))
                if parent[(x, y)] is None:
                    break  # Stop at the start node
                x, y = parent[(x, y)]
            return path[::-1]  # Return reversed path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
                stack.append((nx, ny))
                parent[(nx, ny)] = (x, y)

    return None

class PlannerAgent:
	
	def __init__(self):
		pass
	
	def plan_action(self, world: np.ndarray, current: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[np.ndarray]:
		"""
		Computes a action to take from the current position caputure the pursued while evading from the pursuer

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
		
		directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                  	  		   [-1, -1], [-1, 1], [1, -1], [1, 1]]) 

		return directions[np.random.choice(9)]


