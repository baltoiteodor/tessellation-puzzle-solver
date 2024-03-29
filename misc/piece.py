from misc.types import *


class Piece:
    _numRows: int = 0
    _numCols: int = 0
    _orderNum: int = 0
    _grid: Grid = None
    _rotatable: bool = False

    def __init__(self, grid: Grid):
        self._numRows = len(grid)
        self._numCols = len(grid[0])
        self._orderNum = 0
        self._grid = grid

    def canRotate(self, canRotate: bool):
        self._rotatable = canRotate

    def isRotatable(self):
        return self._rotatable

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

    def __repr__(self):
        return f"Piece {self._orderNum} of size {self._numRows} x {self._numCols}:\n{self._grid}"

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
