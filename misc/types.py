from typing import TypeVar, List

# Types for the project.

Grid = List[List[int]] # Acc an ndarray lmao.
Board = Grid

def prettyPrintGrid(grid: Grid):
    for row in grid:
        print(row)