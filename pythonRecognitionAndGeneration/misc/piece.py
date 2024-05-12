import numpy as np

from misc.contour import Contour
from misc.types import *


class Piece:
    _numRows: int = 0
    _numCols: int = 0
    _orderNum: int = 0
    _grid: Grid = None
    _boardable: bool = False
    _colour: Colour = None
    _numRotations = 4
    _currentRotation = 0
    _allGrids: [Grid] = None
    _rows: [int] = None
    _cols: [int] = None

    def __init__(self, originalContour: Contour, grid: Grid, colour: Colour, unitLen, topLeft):
        self._originalContour = originalContour
        self._numRows = len(grid)
        self._numCols = len(grid[0])
        self._orderNum = 0
        self._grid = grid
        self._allGrids = []
        self._allGrids.append(grid)
        self._rows = []
        self._rows.append(len(grid))
        self._cols = []
        self._cols.append(len(grid[0]))
        self._colour = colour
        self._unitLen = unitLen
        self._topLeft = topLeft
        self._numRotations = 0
        self._currentRotation = 0
        self._currentAngle = 0

    def canBeBoard(self, canBoard: bool, jigsawMode):
        # Here we calculate all possible rotations and how many there are for this piece.
        # Calculate numRotations in general by rotating the piece and add it to the map.
        self._numRotations = 1
        self._boardable = canBoard
        while True:
            self._rotate90()
            if (not jigsawMode) and np.array_equal(self.getGrid(), self._allGrids[0]):
                break
            if jigsawMode and self._currentAngle == 0:
                break
            self._numRotations += 1
            self._allGrids.append(self.getGrid())

        # Clock-wise rotations complete. Next are the horizontal flip + rotations.
        # Available only for tessellation puzzles.
        if not jigsawMode:
            self._flipH()
            if np.array_equal(self.getGrid(), self._allGrids[0]):
                # No more rotations
                self.setGrid(self._allGrids[0])
                return
            rot = self._numRotations
            self._numRotations += 1
            self._allGrids.append(self.getGrid())
            while True:
                self._rotate90()
                if np.array_equal(self.getGrid(), self._allGrids[rot]):
                    break
                self._numRotations += 1
                self._allGrids.append(self.getGrid())

        # Now we have all grids and their number.
        # print(f"For piece {self._grid}, we have rotations: ", self._allGrids)
        self.setGrid(self._allGrids[0])

    # Flips the matrix horizontally.
    def _flipH(self):
        self.setGrid(np.flip(self._grid, axis=1))

    # Rotates piece clock-wise 90 degrees.
    def _rotate90(self):
        rotatedGrid = np.zeros((self.columns(), self.rows()), dtype=int)
        # print(rotatedGrid)
        for i in range(self.rows()):
            for j in range(self.columns()):
                rotatedGrid[j][self.rows() - i - 1] = self.pixelAt(i, j)

        self.setGrid(rotatedGrid)
        oldRows = self.rows()
        oldCols = self.columns()
        self.setRowsNum(oldCols)
        self.setColsNum(oldRows)
        self._currentAngle += 90
        self._currentAngle %= 360

    def rotatePiece(self):
        # Set board to next rotation in line from the array of grids.
        self._currentRotation += 1
        if self._currentRotation == self._numRotations:
            self._currentRotation = 0

        self.setGrid(self._allGrids[self._currentRotation])
        self.setRowsNum(len(self._grid))
        self.setColsNum(len(self._grid[0]))

    def retrieveAngle(self, grid):
        for i, currGrid in enumerate(self._allGrids):
            if np.array_equal(grid, currGrid):
                return i * 90
        return 0

    def getTopLeft(self):
        return self._topLeft

    def getRotations(self):
        return self._numRotations

    def isBoardable(self):
        return self._boardable

    def setOrderNumber(self, num: int):
        self._orderNum = num


    def rows(self):
        return self._numRows

    def columns(self):
        return self._numCols

    # Not very safe, I agree, but it saves a bunch of len() operations.
    def setRowsNum(self, rows: int):
        self._numRows = rows

    def setColsNum(self, cols: int):
        self._numCols = cols

    def pixelAt(self, row: int, col: int):
        return self._grid[row][col]

    def pixelSet(self, row: int, col: int, value: int):
        self._grid[row][col] = value

    def orderNum(self):
        return self._orderNum

    def area(self):
        return self._numCols * self._numRows

    def getGrid(self):
        return self._grid

    def setGrid(self, grid: Grid):
        self._grid = grid

    def getColour(self):
        return self._colour

    def getCurrentAngle(self):
        return self._currentAngle

    # def getColourGrid(self):
    #     return self._colours

    # Each cell in the colour grid will be an average of the colour surrounding
    # the center pixel of the respective cell. This is computed inside the
    # Processor class.
    # def setColourAt(self, row: int, col: int, rgb: Colour):
    #     self._colours[row][col] = rgb
    #
    # def getColourAt(self, row: int, col: int):
    #     return self._colours[row][col]
    def getOriginalContour(self):
        return self._originalContour

    def getUnitLen(self):
        return self._unitLen

    def increaseCurrentRotation(self):
        self._currentRotation += 1
        if self._currentRotation == self._numRotations:
            self._currentRotation = 0

    def __repr__(self):
        return f"Piece {self._orderNum} of size {self._numRows} x {self._numCols} and colour {self._colour}:" \
               f"\n{self._grid}"

    def __eq__(self, other):
        if isinstance(other, Piece):
            if self.area() == other.area() and self.orderNum() == other.orderNum():
                return self._grid == other._grid
            return False
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


# Types for this class.
Pieces = List[Piece]
