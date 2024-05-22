import sys
sys.path.append('..')

from build.DLXCPP import *

def solveDLXCPP(rows, width):
    solver = DLXCPPSolver(rows, width)
    # print("Rows: ", solver.getRows())
    # print(solver.getWidth())
    result = solver.solve(1000)
    print("Result ALOHA: ", len(result))

    return result