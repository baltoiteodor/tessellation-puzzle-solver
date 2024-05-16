import numpy as np

from misc.contour import Contour
from misc.types import *
import cv2
import matplotlib.pyplot as plt

def trimColourGrid(grid):
    # Convert the list of lists to a NumPy array for easier manipulation
    grid = np.array(grid)

    # Assuming that each element in the grid can be compared to (0, 0, 0)
    black = np.array([0, 0, 0], dtype=np.uint8)

    # Remove the last columns if all zero.
    while np.all(np.all(grid[:, -1] == black, axis=1)):
        grid = grid[:, :-1]

    # Remove leading columns with all zeros
    while np.all(np.all(grid[:, 0] == black, axis=1)):
        grid = grid[:, 1:]

    # Remove the last row if all zero.
    while np.all(np.all(grid[-1, :] == black, axis=1)):
        grid = grid[:-1, :]

    # Remove leading rows with all zeros
    while np.all(np.all(grid[0, :] == black, axis=1)):
        grid = grid[1:, :]

    return grid


def similarGrids(grid1, grid2):
    if (len(grid1) != len(grid2)) or (len(grid1[0]) != len(grid2[0])):
        return False

    for r in range(len(grid1)):
        for c in range(len(grid1[0])):
            if (grid1[r][c] == 0 and grid2[r][c] != 0) or (grid1[r][c] != 0 and grid2[r][c] == 0):
                return False
    return True

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

        # Made this for jigsaw colour matching.
        self._colourGrid = []
        self._allColourGrids = []


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

    def _rotateColourGrid(self, rows, columns, index):
        rotatedColouredGrid = [[(0, 0, 0) for _ in range(rows)] for _ in range(columns)]
        # print(rotatedGrid)
        for i in range(rows):
            for j in range(columns):
                rotatedColouredGrid[j][rows - i - 1] = self._allColourGrids[index][i][j]

        self._allColourGrids.append(rotatedColouredGrid)

    def rotatePiece(self):
        # Set board to next rotation in line from the array of grids.
        self._currentRotation += 1
        if self._currentRotation == self._numRotations:
            self._currentRotation = 0

        self.setGrid(self._allGrids[self._currentRotation])
        self._setColourGrid(self._allColourGrids[self._currentRotation])
        self.setRowsNum(len(self._grid))
        self.setColsNum(len(self._grid[0]))

    def retrieveAngle(self, grid):
        for i, currGrid in enumerate(self._allGrids):
            if similarGrids(grid, currGrid):
                return i * 90
        return 0

    def computeColourGrid(self, blurredImage, copyGrid):
        (gridX, gridY) = self._topLeft
        self._colourGrid = [[(0, 0, 0) for _ in range(len(copyGrid[0]))] for _ in range(len(copyGrid))]
        unit = self.getUnitLen()

        # print("contour: ", copyGrid)
        # print("ce plm", len(copyGrid), len(copyGrid[0]), len(self._colourGrid), len(self._colourGrid[0]))
        # print(self._colourGrid)
        # print("top: ", self._topLeft)
        for i in range(len(copyGrid)):
            for j in range(len(copyGrid[0])):
                if copyGrid[i][j] > 0:
                    # print(i, j)
                    # Already in y, x format the grid.
                    topY = gridY + i * unit
                    topX = gridX + j * unit
                    # First point will be the centre of the cell.
                    centreX = topX + (unit / 2)
                    centreY = topY + (unit / 2)
                    # print("centre: ", centreX, centreY)
                    # print(cv2.pointPolygonTest(self._originalContour.getContour(), (int(centreX), int(centreY)), False))
                    # Find 4 other important points that are also inside the contour to average out.
                    dx = []
                    dy = []
                    for px in range(40):
                        for py in range(40):
                            dx.append(px - 20)
                            dy.append(py - 20)
                    # dx = [-10, 0, 10, 0, 5, 5, 0, -5, -5, 0]
                    # dy = [0, 10, 0, -10, 0, 5, 5,  0, -5, -5]
                    points = []
                    points.append((centreX, centreY))
                    for mm in range(len(dx)):
                        points.append((centreX + dx[mm], centreY + dy[mm]))
                    insidePoints = [pt for pt in points if cv2.pointPolygonTest(self._originalContour.getContour(), pt, False) > 0]
                    # print("Inside: ", insidePoints)
                    cv2.imwrite("cema.png", blurredImage)
                    colours = [blurredImage[int(y)][int(x)] for (x, y) in insidePoints]
                    colours = np.array(colours, dtype=np.uint8).reshape(-1, 1, 3)
                    # print("Check  this: ", colours)
                    labColours = cv2.cvtColor(colours, cv2.COLOR_BGR2Lab)
                    # print(labColours)
                    # plt.imshow(labColours)
                    # plt.axis('off')  # Turn off axis labels
                    # plt.show()
                    meanLAB = np.mean(labColours, axis=0)[0]
                    meanLAB = np.uint8([[meanLAB]])
                    # plt.imshow(meanLAB)
                    # plt.axis('off')  # Turn off axis labels
                    # plt.show()

                    # meanRGB = np.mean(colours, axis=0).astype(np.uint8)[0]
                    # meanRGB = np.uint8(meanRGB])
                    # print(meanRGB)
                    meanRGB = cv2.cvtColor(meanLAB, cv2.COLOR_Lab2BGR)[0][0]
                    self._colourGrid[i][j] = meanRGB
        self._colourGrid = trimColourGrid(self._colourGrid)
        self._allColourGrids.append(self._colourGrid)
        rows = len(self._colourGrid)
        cols = len(self._colourGrid[0])
        # print(rows, cols)
        self._rotateColourGrid(rows, cols, 0)
        self._rotateColourGrid(cols, rows, 1)
        self._rotateColourGrid(rows, cols, 2)

        # print("fmmmmmmm?: ", self._allColourGrids)
        # for c in self._allColourGrids:
        #     plt.imshow(c)
        #     plt.axis('off')  # Turn off axis labels
        #     plt.show()
        # Rotate this bad boy 3 more times and add to allColourGrids.

        # Uncomment this if want to see how colours of pieces look. Okish for testing, might look weird bcuz of LAB.
        # plt.imshow(self._colourGrid)
        # plt.axis('off')  # Turn off axis labels
        # plt.show()
        # pass

    def showColour(self):
        plt.imshow(self._colourGrid)
        plt.axis('off')  # Turn off axis labels
        plt.show()
    def getColourGrid(self):
        return self._colourGrid

    def _setColourGrid(self, colourGrid):
        self._colourGrid = colourGrid
    def getColourAt(self, row, col):
        return self._colourGrid[row][col]

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
