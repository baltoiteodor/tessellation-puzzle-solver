from puzzle_solver.DLX.MatrixElement import MatrixElement


# List headers have the same fields as a matrix element plus a size field and a name field.
class ListHeader(MatrixElement):

    def __init__(self, idElem: int):
        super().__init__(idElem)
        self._C = self
        self._S = 0
        self._N = None

    def setSize(self, size: int):
        self._S = size

    def setName(self, name):
        self._N = name

    def size(self):
        return self._S

    def name(self):
        return self._N




