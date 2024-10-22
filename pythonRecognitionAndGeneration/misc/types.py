from typing import TypeVar, List

# Types for the project.

Grid = List[List[int]]
Board = Grid
Colour = (int, int, int)


def prettyPrintGrid(grid: Grid):
    for row in grid:
        print(row)
