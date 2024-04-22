from puzzle_solver.DLX.ListHeader import ListHeader
from puzzle_solver.DLX.MatrixElement import MatrixElement


# Class for the master header.
class DLXMatrix(ListHeader):

    # Will keep a list that maps ids of objects to themselves.
    _objects: [MatrixElement] = []
    # The id the next MatrixElement created will have and also the size of the _objects list.
    _lastID: int = 0
    _root: int

    def createElement(self):
        newElem = MatrixElement(self._lastID)
        self._objects.append(newElem)
        self._lastID += 1
        return self._lastID - 1

    # Given a 2d array consisting of the boolean rows representing the exact cover problem
    def createDLXMatrix(self, rows, boardSize: int, numPieces: int):
        # Create the root first.
        root = self.createElement()

        # Add the columns next, a total of boardSize + numPieces.
