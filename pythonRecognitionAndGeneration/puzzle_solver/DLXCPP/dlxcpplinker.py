import sys

sys.path.append('..')

from build.DLXCPP import *


def solveDLXCPP(rows, width, jigsaw):
    solver = DLXCPPSolver(rows, width)
    if jigsaw:
        # Stop after 25k solutions found.
        result = solver.solve(25000)
    else:
        result = solver.solve(1)
    print("Number of potential solutions found: ", len(result))

    return result
