from build.DLXCPP import *

print(add_cpp(3, 5))

solver = DLXCPPSolver(5)

print(solver.printer())

solver.addRows([[1, 2, 3], [1, 2, 3]])

print(solver.getRows())

