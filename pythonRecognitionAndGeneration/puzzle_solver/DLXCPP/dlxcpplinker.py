import sys
sys.path.append('..')

from build.DLXCPP import *

def solveDLXCPP(rows, width):
    solver = DLXCPPSolver(rows, width)
    print("HEHEHEEEEEEEE: ", solver.getRows())
    print(solver.getWidth())
    result = solver.solve()
    print("Lmao? ", result)

    return result