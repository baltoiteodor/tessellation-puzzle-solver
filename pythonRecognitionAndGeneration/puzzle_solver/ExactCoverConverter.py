# This class converts a list of pieces and a board to a matrix representing an exact cover
# problem.
import numpy as np

from misc.piece import Piece
from puzzle_solver.helper import similarColours, similarColoursJigsaw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import cv2

# 1
FAULTYNUMX = 1
# 0
FAULTYNUMTHUMB = 1
# 1
FAULTYNUMHOLES = 1
# 36
COLOURTHRESHOLDX = 36
# 42
COLOURTHRESHOLDTHUMB = 42
# 38
COLOURTHRESHOLDHOLE = 38

class ExactCoverConverter:
    def __init__(self, board: Piece, pieces, colourMap, version, colour, jigsaw):
        # Board will be the biggest piece filled with 1s. We shape the matrix w.r.t it.
        self._pypyColumns = None
        self._pypyRows = None
        self._boardPiece = board
        self._pieces = pieces
        self._matrix = None
        self._rows = 0
        self._cols = 0
        # If version is 0, use home-made DLX, otherwise use the pypy one.
        self._version = version
        # If this is enabled, we check for colour information to match as well.
        self._colouring = colour
        self._colourMap = colourMap
        self._debugChecks = 0
        self._time = 0
        self._colourPairsDictionary = {}
        # For jigsaw:
        self._jigsawTwosBoard = np.zeros((board.rows(), board.columns()))
        self._jigsaw = jigsaw
        self._availablePositionsPerPiece = {}
        for piece in pieces:
            self._availablePositionsPerPiece[piece.orderNum()] = []

        # Adaptive thresholds set for colour matching.
        self._FAULTYNUMX = FAULTYNUMX
        self._FAULTYNUMTHUMB = FAULTYNUMTHUMB
        self._FAULTYNUMHOLES = FAULTYNUMHOLES
        self._COLOURTHRESHOLDX = COLOURTHRESHOLDX
        self._COLOURTHRESHOLDTHUMB = COLOURTHRESHOLDTHUMB
        self._COLOURTHRESHOLDHOLE = COLOURTHRESHOLDHOLE

    def setThresholds(self, FAULTYNUMX, FAULTYNUMTHUMB, FAULTYNUMHOLES,
                      COLOURTHRESHOLDX, COLOURTHRESHOLDTHUMB, COLOURTHRESHOLDHOLE):

        self._FAULTYNUMX = FAULTYNUMX
        self._FAULTYNUMTHUMB = FAULTYNUMTHUMB
        self._FAULTYNUMHOLES = FAULTYNUMHOLES
        self._COLOURTHRESHOLDX = COLOURTHRESHOLDX
        self._COLOURTHRESHOLDTHUMB = COLOURTHRESHOLDTHUMB
        self._COLOURTHRESHOLDHOLE = COLOURTHRESHOLDHOLE
        self._availablePositionsPerPiece = {}
        for piece in self._pieces:
            self._availablePositionsPerPiece[piece.orderNum()] = []



    def constructMatrix(self):
        # The width of the DLX matrix will be the size of the board piece for information on
        # what positions are taken, to which we add the number of pieces.
        boardSize = self._boardPiece.columns() * self._boardPiece.rows()
        piecesNum = len(self._pieces)
        width = boardSize + piecesNum
        self._matrix = []
        self._cols = width
        # Each piece will contribute a number of rows equal to the number of possible
        # rotations it can take * all valid locations in can be placed.

        # This is for PyPy dlx:
        self._pypyColumns = []
        self._pypyRows = []
        if self._version:
            for i in range(self._boardPiece.rows()):
                for j in range(self._boardPiece.columns()):
                    self._pypyColumns.append("r" + str(i) + "c" + str(j))
                    if self._jigsaw:
                        # Draw the Twos matrix, a.k.a where the X shape of each piece can go.
                        if i % 3 == 0:
                            if j % 3 == 0 or j % 3 == 2:
                                self._jigsawTwosBoard[i][j] = 2
                        elif i % 3 == 1 and j % 3 == 1:
                            self._jigsawTwosBoard[i][j] = 2
                        elif i % 3 == 2:
                            if j % 3 == 0 or j % 3 == 2:
                                self._jigsawTwosBoard[i][j] = 2
            # print("Tuturuuuu: ", self._jigsawTwosBoard)
            for i in range(piecesNum):
                self._pypyColumns.append("P" + str(i + 1))

        if self._version == 1:
            self.PyPyMatrixNotOptimised(boardSize, width)
        elif self._version == 2:
            self.PyPyMatrixOptimised(boardSize, width)
        print(self._COLOURTHRESHOLDHOLE, self._COLOURTHRESHOLDX)
        return width

    def PyPyMatrixOptimised(self, boardSize, width):
        for r in range(self._boardPiece.rows()):
            for c in range(self._boardPiece.columns()):
                # For each position in the board, check what piece might fit.
                for piece in self._pieces:
                    # Check rotations.
                    numRotations = piece.getRotations()
                    for rot in range(numRotations):
                        # if piece.orderNum() == 1:
                        #     piece.showColour()
                        #     print(piece.getGrid())
                        #     cv2.imwrite("ok.png", piece.getOriginalContour().getImage())
                        #     plt.imshow(self._colourMap)
                        #     plt.axis('off')  # Turn off axis labels
                        #     plt.show()

                        pypyRow = []
                        if self._checkOptimised(r, c, piece, pypyRow):
                            pypyRow.append(boardSize - 1 + piece.orderNum())
                            self._pypyRows.append(pypyRow)
                            self._availablePositionsPerPiece[piece.orderNum()].append((r, c, rot))
                            # if piece.orderNum() == 6 and r == 0 and c == 9:
                            #     print("Bitch worked here btw: ", piece.orderNum(), r, c)
                            #     piece.showColour()
                        # Rotate piece.
                        piece.rotatePiece()
                        # if piece.orderNum() == 6 and r == 0 and c == 9:
                        #     piece.showColour()
                        #     print(piece.getGrid())


    def _checkOptimised(self, row, col, piece, pypyRow):
        if row + piece.rows() - 1 >= self._boardPiece.rows() or (
                col + piece.columns() - 1 >= self._boardPiece.columns()):
            return False
        # For jigsaw check if more than 2 out of 5 cells are wrong to return false.
        faulty = 0
        faultyThumbs = 0
        faultyHoles = 0
        # colourDist = 0
        for r in range(row, row + piece.rows()):
            for c in range(col, col + piece.columns()):
                self._debugChecks += 1

                # Rule out cases where Piece is in an unfeasible location.
                if self._jigsaw and piece.pixelAt(r - row, c - col) == 1 and self._jigsawTwosBoard[r][c] == 2:
                    return False

                # Check for Edges first.

                if self._jigsaw and piece.pixelAt(r - row, c - col) == 0 and self._jigsawTwosBoard[r][c] == 0:
                    if r == 0 or c == 0 or r == self._boardPiece.rows() - 1 or c == self._boardPiece.columns() - 1:
                        return False

                timeIn = timer()
                # If jigsaw and we deal with a hole.
                # print(piece.getColourAt(r - row, c - col))
                if self._colouring and self._jigsaw and (piece.pixelAt(r - row, c - col) == 0 and self._jigsawTwosBoard[r][c] == 0 and not np.array_equal(piece.getColourAt(r - row, c - col), [0, 0, 0])
                                                         and not (similarColoursJigsaw(piece.getColourAt(r - row, c - col), self._colourMap[r][c], self._colourPairsDictionary, self._COLOURTHRESHOLDHOLE))):
                    timeOut = timer()
                    self._time += timeOut - timeIn
                    faultyHoles += 1
                    # print("This happened: ", piece.orderNum(), piece.getColourAt(r - row, c - col), r - row, c - col)
                # If jigsaw and we deal with a thumb.
                if self._colouring and self._jigsaw and (piece.pixelAt(r - row, c - col) == 1 and self._jigsawTwosBoard[r][c] == 0 and
                                                         not (similarColoursJigsaw(piece.getColourAt(r - row, c - col), self._colourMap[r][c], self._colourPairsDictionary, self._COLOURTHRESHOLDTHUMB))):
                    timeOut = timer()
                    self._time += timeOut - timeIn
                    faultyThumbs += 1
                # print(r, c, r - row, c - col, piece.pixelAt(r - row, c - col) == 2, similarColoursJigsaw(piece.getColourAt(r - row, c - col), self._colourMap[r][c], self._colourPairsDictionary))
                if self._colouring and self._jigsaw and (piece.pixelAt(r - row, c - col) == 2 and self._jigsawTwosBoard[r][c] == 2 and
                                        not (similarColoursJigsaw(piece.getColourAt(r - row, c - col), self._colourMap[r][c], self._colourPairsDictionary, self._COLOURTHRESHOLDX))):
                    # print("tried and not worked: ", piece.orderNum(), r - row, c - col, r, c)
                    # print(piece.getColourAt(r - row, c - col))
                    # print(self._colourMap[r][c])
                    # plt.imshow(fmm)
                    # plt.axis('off')  # Turn off axis labels
                    # plt.show()
                    timeOut = timer()
                    self._time += timeOut - timeIn
                    faulty += 1
                    # return False
                # print("sa moara mata")
                if self._colouring and not self._jigsaw and (piece.pixelAt(r - row, c - col) != 0 and
                                        not similarColoursJigsaw(piece.getColour(), self._colourMap[r][c], self._colourPairsDictionary, self._COLOURTHRESHOLDX)):
                    timeOut = timer()
                    self._time += timeOut - timeIn
                    return False
                # print("si tactu")
                if piece.pixelAt(r - row, c - col):
                    pypyRow.append(r * self._boardPiece.columns() + c)

                # colourDist += similarColoursJigsaw(piece.getColourAt(r - row, c - col), self._colourMap[r][c], self._colourPairsDictionary)
        if self._jigsaw:
            # print(faulty)
            if faulty > self._FAULTYNUMX or faultyThumbs > self._FAULTYNUMTHUMB or faultyHoles > self._FAULTYNUMHOLES:
                return False
                # or colourDist > COLOURSUMTHRESHOLD):
        return True

    def PyPyMatrixNotOptimised(self, boardSize, width):
        for piece in self._pieces:
            numRotations = piece.getRotations()
            for _ in range(numRotations):
                # Current rotation of piece.
                for row in range(self._boardPiece.rows() - piece.rows() + 1):
                    for col in range(self._boardPiece.columns() - piece.columns() + 1):
                        matrixRow = np.zeros(width, dtype=bool)
                        pypyRow = []
                        if self._checkPlace(self._boardPiece.columns(), row, col, piece, matrixRow, pypyRow):
                            # Add a 1 in the matrixRow at the piece orderNum location.
                            matrixRow[boardSize - 1 + piece.orderNum()] = 1
                            if self._version:
                                pypyRow.append(boardSize - 1 + piece.orderNum())
                                self._pypyRows.append(pypyRow)
                            # Add it to the matrix.
                            self._matrix.append(matrixRow)
                            self._rows += 1

                # Rotate piece.
                piece.rotatePiece()

    def _checkPlace(self, boardColumns: int, row: int, col: int, piece, matrixRow, pypyRow):
        for r in range(row, row + piece.rows()):
            for c in range(col, col + piece.columns()):
                self._debugChecks += 1
                if self._colouring and (piece.pixelAt(r - row, c - col) != 0 and
                                        not similarColours(piece.getColour(), self._colourMap[r][c], self._colourPairsDictionary)):
                    return False
                if piece.pixelAt(r - row, c - col):
                    # Means colour must be similar.
                    matrixRow[r * boardColumns + c] = 1
                    if self._version:
                        pypyRow.append(r * boardColumns + c)
        return True

    def printMatrix(self):
        if self._version:
            print(f"Debug checks: {self._debugChecks}.")
            print("Columns: ", self._pypyColumns)
            print("Time spent colour checking: ", self._time)
            for piece in self._pieces:
                print(f"Available for piece {piece.orderNum()}: ", self._availablePositionsPerPiece[piece.orderNum()])
            print(f"These rows be for pypy bro: {len(self._pypyRows)}")
            return

        # print("Matrix before constructing dancing links.")
        # for row in range(self._rows):
        #     print([int(elem) for elem in self._matrix[row]])

    def getPyPy(self):
        return self._pypyColumns, self._pypyRows

    def getDict(self):
        return self._colourPairsDictionary
    # def _constructMatrixPyPy(self):
    #     boardSize = self._boardPiece.columns() * self._boardPiece.rows()
    #     width = boardSize + len(self._pieces)
