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
	
	def plan_action(self, world: np.ndarray, current: Tuple[int, int], pursued: Tuple[int, int], pursuer: Tuple[int, int]) -> Optional[np.ndarray]:
		"""
		Computes a path from the start position to the end position 
		using a certain planning algorithm (DFS is provided as an example).

		Parameters:
		- world (np.ndarray): A 2D numpy array representing the grid environment.
		- 0 represents a walkable cell.
		- 1 represents an obstacle.
		- start (Tuple[int, int]): The (row, column) coordinates of the starting position.
		- end (Tuple[int, int]): The (row, column) coordinates of the goal position.

		Returns:
		- np.ndarray: A 2D numpy array where each row is a (row, column) coordinate of the path.
		The path starts at 'start' and ends at 'end'. If no path is found, returns None.
		"""
		
		directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                  	  		   [-1, -1], [-1, 1], [1, -1], [1, 1]]) 

		return directions[np.random.choice(9)]


