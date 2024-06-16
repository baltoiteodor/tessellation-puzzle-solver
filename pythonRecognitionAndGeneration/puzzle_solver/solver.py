from matplotlib import pyplot as plt

from puzzle_solver.DLXPyPy import DLX
from puzzle_solver.DLXCPP import dlxcpplinker
from puzzle_solver.ExactCoverConverter import ExactCoverConverter
from puzzle_solver.helper import *
from timeit import default_timer as timer
import numpy as np
import cv2
import copy

# Initial thresholds for colour matching jigsaw pieces.
# 1
FAULTYNUMX = 0
# 0
FAULTYNUMTHUMB = 1
# 1
FAULTYNUMHOLES = 1
# 36
COLOURTHRESHOLDX = 20
# 42
COLOURTHRESHOLDTHUMB = 30
# 38
COLOURTHRESHOLDHOLE = 40
def arrayToTuple(array):
    return tuple(tuple(row) for row in array)

class Solver:
    def __init__(self, logger: bool, image, bkt: int, dlx: int, colour: bool, cpp: bool, jigsaw: bool,
                 scalingDisabled: bool):
        self._logger = logger
        self._startTime = self._endTime = 0
        self._solution = None
        self._image = image
        self._bkt = bkt
        self._dlx = dlx
        self._colourMatters = colour
        self._used = 0
        self._sol = []
        self._size = 0
        self._cpp = cpp
        self._jigsaw = jigsaw
        self._dictIndexToPiece = {}
        self._colourMap = []
        self._nonScaling = scalingDisabled
        self._boardPiece = None
        # For jigsaw mode, all possible solutions to the puzzle.
        self._allSolutions = []
        self._bktOutcome = False
        self._bktAllSolutions = []
        self._alreadySol = {}
        self._colourPairsDictionary = {}


    # The solve method will take as input an array of 2d arrays representing puzzle pieces and try to solve the puzzle.
    def solveBackTracking(self, pieces: Pieces):
        self._startTime = timer()
        self._firstSolTime = timer()
        if self._logger:
            print("Entering Solver class...")

        boardPiece = findBoard(pieces)
        board = boardPiece.getGrid()

        initialBoardPiece = boardPiece
        ##########
        # Works for 2x! Somehow.
        # verdict, boardPiece = scalePiece(boardPiece, 2.0, self._image)

        ###########

        # Add an order number such that we can differentiate pieces in the output matrix.
        # piecesIndex = [(x, index + 1) for index, x in enumerate(pieces)]
        for index, piece in enumerate(pieces):
            # if index == 0:
            # cv2.imwrite(f"piece{index}.png", piece.getOriginalContour().getImage())
            # plt.imshow(piece.getColourGrid())
            # plt.axis('off')  # Turn off axis labels
            # plt.show()
            piece.setOrderNumber(index + 1)

        # Calculate real area of pieces to determine scale factor of board.
        if not self._jigsaw:
            piecesArea = calculatePiecesArea(pieces)
            boardArea = initialBoardPiece.getOriginalContour().getArea()
            scaler = np.sqrt(piecesArea / boardArea)
            # print(scaler)
            scaler = roundScaler(scaler)
            if self._nonScaling:
                # In this case we are not allowed to scale:
                if scaler != 1:
                    print(
                        "The puzzle cannot be solved as the area of the pieces and the area of the board do not correspond.")
                    return False

            if self._logger:
                print("Scaler: ", scaler)

            verdict = True
            if scaler != 1.0:
                verdict, boardPiece = scalePiece(boardPiece, scaler, self._image)

            if verdict == False:
                print("Scaling the board makes the puzzle unsolvable.")
                return False

        # print("Board Piece: ", boardPiece)
        # In BGR format.
        if not self._jigsaw:
            self._extractColourMap(boardPiece)
        else:
            self._colourMap = boardPiece.getColourGrid()

        if self._logger:
            print(f"Puzzle Board:")
            print(board)
            print("Colour map for board:\n", self._colourMap)
            plt.imshow(self._colourMap)
            plt.axis('off')  # Turn off axis labels
            plt.show()
        self._boardPiece = boardPiece
        # Remember colours of pieces.
        piecesColour = [pc.getColour() for pc in pieces]
        # We will start the bkt from empty board.
        startingBoard = emptyBoard(boardPiece.rows(), boardPiece.columns())
        outputMatrix = emptyBoard(boardPiece.rows(), boardPiece.columns())

        if self._logger:
            print("Indexed pieces: ")
            print(pieces)

        for piece in pieces:
            self._dictIndexToPiece[piece.orderNum()] = piece

        self._size = len(pieces)
        self._sol = [0 for _ in range(self._size)]
        outcome = None
        if self._bkt == 0:
            outcome = self._backtrackNoOptimisation(startingBoard, board, outputMatrix, pieces, 0, 0)
        elif self._bkt == 1:
            self._backtrackRotationOptimisation(startingBoard, board, outputMatrix, pieces, 0, 0)
            outcome = self._bktOutcome
            self._endTime = timer()
            if len(self._bktAllSolutions) != 0:
                outputMatrix = self._bktAllSolutions[0]
                # print(len(self._bktAllSolutions))
                if self._logger:
                    print("Puzzle completed successfully.")
                    print(f"Exiting Solver class: {self._endTime - self._startTime}...")
                    print("---")
                    print("----------------------------")
                    print("---")
        elif self._dlx != -1:
            outcome = self._DLX(boardPiece, pieces, self._dlx, outputMatrix)

        self._output = outputMatrix
        if outcome:
            # Construct matrix with colours.
            self._solution = [[(0.0, 0.0, 0.0) for _ in range(boardPiece.columns())] for _ in range(boardPiece.rows())]

            if self._logger:
                print("Solution in the piece form: ")
                prettyPrintGrid(outputMatrix)
            for i in range(boardPiece.rows()):
                for j in range(boardPiece.columns()):
                    indexPiece = outputMatrix[i][j]
                    self._solution[i][j] = piecesColour[indexPiece - 1]
        return outcome

    # TODO: add flips as well.
    def _backtrackNoOptimisation(self, currentBoard: Board, board: Board, outputMatrix: Board,
                                 pieces: Pieces, currRow: int, currCol: int):
        # Get the first 0 in the grid as updated position.
        row, col = nextPos(currentBoard, currRow, currCol)
        if self._used == self._size and row == -1 and col == -1:  # A.k.a. outside the grid and no pieces left.
            self._endTime = timer()
            if self._logger:
                print("Puzzle completed successfully.")
                print(f"Exiting Solver class: {self._endTime - self._startTime}...")
                print("---")
                print("----------------------------")
                print("---")
            return True

        # Try all possible pieces (with different rotations as well).
        for i in range(self._size):
            piece = pieces[i]
            if self._sol[i] == 1:
                # Already in the solution
                continue
            # Move to next position with the remaining pieces if at least one rotation is valid.
            for _ in range(4):
                decision, newRow, newCol = isValid(currentBoard, board, self._colourMap,
                                                   piece, row, col, self._colourMatters)
                # if self._logger:
                #     print(f"Trying piece {piece} in position {row, col}. \nWIth current board")
                #     prettyPrintGrid(currentBoard)
                if decision:
                    setPiece(currentBoard, board, outputMatrix, piece, newRow, newCol)
                    # if len(pieces) >= 3:
                    #     print("Heh")
                    #     prettyPrintGrid(outputMatrix)
                    # Remove from list of pieces.
                    # pieces.remove(piece)
                    self._used += 1
                    self._sol[i] = 1

                    if self._backtrackNoOptimisation(currentBoard, board, outputMatrix, pieces, newRow, newCol):
                        return True

                    self._used -= 1
                    self._sol[i] = 0
                    # pieces.append(piece)

                    # Backtrack, remove the piece.
                    removePiece(currentBoard, piece, newRow, newCol)
                rotatePieceNonOptimal(piece)
        return False

    # Pre-calculates all rotations.

    def _backtrackRotationOptimisation(self, currentBoard: Board, board: Board, outputMatrix: Board,
                                       pieces: Pieces, currRow: int, currCol: int):
        # # TODO: Del this
        # if self._bktOutcome:
        #     return
        # Get the first 0 in the grid as updated position.
        row, col = nextPos(currentBoard, currRow, currCol)
        if self._used == self._size and row == -1 and col == -1:  # A.k.a. outside the grid and no pieces left.
            if not self._bktOutcome:
                self._firstSolTime = timer()
            self._bktOutcome = True
            solTuple = arrayToTuple(outputMatrix)
            if solTuple not in self._alreadySol:
                copyM = copy.deepcopy(outputMatrix)
                self._bktAllSolutions.append(copyM)
                self._alreadySol[solTuple] = True
            # return
        # print("sol")
            # prettyPrintGrid(outputMatrix)
            # if self._logger:
            #     print("Puzzle completed successfully.")
            #     print(f"Exiting Solver class: {self._endTime - self._startTime}...")
            #     print("---")
            #     print("----------------------------")
            #     print("---")

        # Try all possible pieces (with different rotations as well).
        for i in range(self._size):
            piece = pieces[i]
            if self._sol[i] == 1:
                # Already in the solution
                continue
            # Move to next position with the remaining pieces if at least one rotation is valid.
            rgLen = self._pickRangeOptimiser(piece)
            # if piece.orderNum() == 1:
            # print("ALEEE", rgLen)
            for _ in range(rgLen):
                decision, newRow, newCol = isValid(currentBoard, board, self._colourMap,
                                                   piece, row, col, self._colourMatters, self._colourPairsDictionary)
                if decision:
                    # print("before")
                    # prettyPrintGrid(outputMatrix)
                    setPiece(currentBoard, board, outputMatrix, piece, newRow, newCol)
                    # print("a")
                    # prettyPrintGrid(outputMatrix)
                    self._sol[i] = 1
                    self._used += 1
                    self._backtrackRotationOptimisation(currentBoard, board, outputMatrix, pieces, newRow, newCol)
                    self._sol[i] = 0
                    self._used -= 1
                    # Backtrack, remove the piece.
                    removePiece(currentBoard, outputMatrix, piece, newRow, newCol)
                    # print("b")
                    # prettyPrintGrid(outputMatrix)
                rotatePiece(piece)

    def adaptiveThresh(self, converter, labels, rows, boardPiece, outputMatrix, width, COLOURTHRESHOLDX,
                       COLOURTHRESHOLDTHUMB, COLOURTHRESHOLDHOLE):
        # converter.setThresholds(0, 1, 1, COLOURTHRESHOLDX, COLOURTHRESHOLDTHUMB, COLOURTHRESHOLDHOLE)
        # width = converter.constructMatrix()
        # labels, rows = converter.getPyPy()
        outcome = self._solveDLXCPP(labels, rows, boardPiece, outputMatrix, width)
        if 150 > outcome > 0:
            return outcome
        threshX = COLOURTHRESHOLDX
        threshThumb = COLOURTHRESHOLDTHUMB
        threshHole = COLOURTHRESHOLDHOLE

        threshXBest = COLOURTHRESHOLDX
        threshThumbBest = COLOURTHRESHOLDTHUMB
        threshHoleBest = COLOURTHRESHOLDHOLE

        minOutcome = 11000
        while outcome > 400:
            threshX -= 3
            threshThumb -= 3
            threshHole -= 3
            converter.setThresholds(0, 1, 1, threshX, threshThumb, threshHole)
            width = converter.constructMatrix()
            labels, rows = converter.getPyPy()
            outcome = self._solveDLXCPP(labels, rows, boardPiece, outputMatrix, width)
            if minOutcome > outcome > 0:
                minOutcome = outcome
                threshXBest = threshX
                threshThumbBest = threshThumb
                threshHoleBest = threshHole
            if 150 > outcome > 0:
                return outcome

        lastTestNum = 10
        good = False
        for it in range(30):
            if outcome == 0:
                # Maybe time it by an increasing factor?
                if not good:
                    threshX += 2
                    threshThumb += 2
                    threshHole += 2
                else:
                    threshThumb += 1
                    threshHole += 1
            else:
                good = True
                if outcome >= 250:
                    threshX -= 1
                    threshThumb -= 2
                    threshHole -= 2
                else:
                    threshX -= 0
                    threshThumb -= 1
                    threshHole -= 1
            if good:
                lastTestNum -= 1
            if lastTestNum == 0:
                break
            converter.setThresholds(1, 1, 1, threshX, threshThumb, threshHole)
            width = converter.constructMatrix()
            labels, rows = converter.getPyPy()
            outcome = self._solveDLXCPP(labels, rows, boardPiece, outputMatrix, width)

            if minOutcome > outcome > 0:
                minOutcome = outcome
                threshXBest = threshX
                threshThumbBest = threshThumb
                threshHoleBest = threshHole

            if 150 > outcome > 0:
                return outcome

        # print("Best stuff: ", threshXBest, threshThumbBest, threshHoleBest)
        converter.setThresholds(1, 1, 1, threshXBest, threshThumbBest, threshHoleBest)
        width = converter.constructMatrix()
        labels, rows = converter.getPyPy()
        outcome = self._solveDLXCPP(labels, rows, boardPiece, outputMatrix, width)

        return outcome

    def _DLX(self, boardPiece: Piece, pieces: Pieces, version, outputMatrix):
        converter = ExactCoverConverter(boardPiece, pieces, self._colourMap, version, self._colourMatters, self._jigsaw)
        if self._cpp:
            converter.setThresholds(0, 1, 1, COLOURTHRESHOLDX, COLOURTHRESHOLDTHUMB, COLOURTHRESHOLDHOLE)
        width = converter.constructMatrix()
        if self._logger:
            converter.printMatrix()
        outcome = 0
        # Version 0 is for homemade python DLX (WIP).
        if (not self._cpp) and version:
            labels, rows = converter.getPyPy()
            outcome = self._solveDLXPyPy(labels, rows, boardPiece, outputMatrix)

        if self._cpp:
            labels, rows = converter.getPyPy()
            # Perform adaptive thresholding.
            if self._jigsaw:
                outcome = self.adaptiveThresh(converter, labels, rows, boardPiece, outputMatrix, width, COLOURTHRESHOLDX,
                                              COLOURTHRESHOLDTHUMB, COLOURTHRESHOLDHOLE)
                if outcome > 11000:
                    return 0
            else:
                outcome = self._solveDLXCPP(labels, rows, boardPiece, outputMatrix, width)

        self._endTime = timer()
        self._dictOfColours = converter.getDict()
        if outcome and self._logger:
            print("Puzzle completed successfully.")
            print(f"Exiting Solver class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")

        return outcome

    def _solveDLXPyPy(self, labels, rows, boardPiece, outputMatrix):
        # labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        # rows = [[0, 3, 6], [0, 3], [3, 4, 6], [2, 4, 5], [1, 2, 5, 6], [1, 6]]
        instance = DLX.genInstance(labels, rows)

        selected = DLX.solveInstance(instance)
        # DLX.printColumnsPerRow(instance, selected)

        # Construct outputMatrix.
        boardSize = boardPiece.rows() * boardPiece.columns()
        _, _, col, _ = instance
        if selected is None:
            return False
        for row in selected:
            # For each selected row, place it in the output matrix.
            chosenPiece = rows[row][-1] - boardSize + 1
            for element in rows[row][:-1]:
                r = int(element / boardPiece.columns())
                c = element % boardPiece.columns()
                outputMatrix[r][c] = chosenPiece
        if self._logger:
            prettyPrintGrid(outputMatrix)

        return True

    def _solveDLXCPP(self, labels, rows, boardPiece, outputMatrix, width):
        # labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        # rows = [[0, 3, 6], [0, 3], [3, 4, 6], [2, 4, 5], [1, 2, 5, 6], [1, 6]]
        timeIn = timer()
        selected = dlxcpplinker.solveDLXCPP(rows, width, self._jigsaw)
        if selected is None or len(selected) == 0:
            return 0

        # if not self._jigsaw:
        #     selected = selected[0]
        # Construct first general outputMatrix.
        boardSize = boardPiece.rows() * boardPiece.columns()
        for row in selected[0]:
            # For each selected row, place it in the output matrix.
            chosenPiece = rows[row][-1] - boardSize + 1
            for element in rows[row][:-1]:
                r = int(element / boardPiece.columns())
                c = element % boardPiece.columns()
                outputMatrix[r][c] = chosenPiece

        timeOut = timer()
        if self._logger:
            print("CPP DLX solver took: ", timeOut - timeIn)
            prettyPrintGrid(outputMatrix)

        # If in jigsaw mode, construct all output matrices.
        if self._jigsaw:
            self._allSolutions = []
            for select in selected:
                currSol = emptyBoard(boardPiece.rows(), boardPiece.columns())
                for row in select:
                    # For each selected row, place it in the output matrix.
                    chosenPiece = rows[row][-1] - boardSize + 1
                    for element in rows[row][:-1]:
                        r = int(element / boardPiece.columns())
                        c = element % boardPiece.columns()
                        currSol[r][c] = chosenPiece
                self._allSolutions.append(currSol)

        return len(selected)

    def _pickRangeOptimiser(self, piece):
        if self._bkt == 0:
            return 4
        return piece.getRotations()

    def _extractColourMap(self, board: Piece):
        self._colourMap = [[(0.0, 0.0, 0.0) for _ in range(board.columns())] for _ in range(board.rows())]
        unit = board.getUnitLen()
        (topLeftX, topLeftY) = board.getTopLeft()
        image = board.getOriginalContour().getImage()
        # Walk again through image and for each 1 in the grid, put the colour of the center or some average.
        # For the 0s put (0,0,0) in colour.
        for i in range(board.rows()):
            for j in range(board.columns()):
                # Already in y, x format the grid.
                topY = topLeftY + i * unit
                topX = topLeftX + j * unit
                centreX = topX + (unit / 2)
                centreY = topY + (unit / 2)
                # Get the colour from original image and put it in the map.
                if board.pixelAt(i, j) != 0:
                    b, g, r = image[int(centreY)][int(centreX)]
                    self._colourMap[i][j] = (r, g, b)

    def getSolution(self):
        return self._solution

    def getDictPieces(self):
        return self._dictIndexToPiece

    def getOutput(self):
        return self._output

    def getBoardPiece(self):
        return self._boardPiece

    def getColourMap(self):
        return self._colourMap

    def getAllSolutions(self):
        return self._allSolutions

    def getTimeTaken(self):
        return self._endTime - self._startTime

    def getColourDict(self):
        return self._dictOfColours

    def getAllBKT(self):
        return self._bktAllSolutions

    def getFirstSolTime(self):
        return self._firstSolTime - self._startTime