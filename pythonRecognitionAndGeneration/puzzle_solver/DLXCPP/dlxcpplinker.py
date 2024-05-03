import sys
sys.path.append('..')

from build.DLXCPP import *

def solveDLXCPP(rows, width):
    solver = DLXCPPSolver(rows, width)
    # print("Rows: ", solver.getRows())
    # print(solver.getWidth())
    result = solver.solve()
    print("Result: ", result)

    return result