from matplotlib import pyplot as plt

from puzzle_solver.helper import *
from timeit import default_timer as timer


class Solver:
    def __init__(self, logger: bool, image, optimise: int):
        self._logger = logger
        self._startTime = self._endTime = 0
        self._solution = None
        self._image = image
        self._optimisationMode = optimise

    # The solve method will take as input an array of 2d arrays representing puzzle pieces and try to solve the puzzle.
    def solveBackTracking(self, pieces: Pieces):

        if self._logger:
            self._startTime = timer()
            print("Entering Solver class...")
            print("Attempting to solve the puzzle using basic backtracking algorithm.")

        boardPiece = findBoard(pieces)
        board = boardPiece.getGrid()
        # In BGR format.
        self._extractColourMap(boardPiece)
        if self._logger:
            print(f"Puzzle Board:")
            print(board)
            print("Colour map for board:\n", self._colourMap)
            # plt.imshow(self._colourMap)
            # plt.axis('off')  # Turn off axis labels
            # plt.show()

        # Add an order number such that we can differentiate pieces in the output matrix.
        # piecesIndex = [(x, index + 1) for index, x in enumerate(pieces)]
        for index, piece in enumerate(pieces):
            piece.setOrderNumber(index + 1)
        # Remember colours of pieces.
        piecesColour = [pc.getColour() for pc in pieces]
        # We will start the bkt from empty board.
        startingBoard = emptyBoard(boardPiece.rows(), boardPiece.columns())
        outputMatrix = emptyBoard(boardPiece.rows(), boardPiece.columns())

        if self._logger:
            print("Indexed pieces: ")
            print(pieces)

        if self._optimisationMode == 0:
            outcome = self._backtrackNoOptimisation(startingBoard, board, outputMatrix, pieces, 0, 0)
        elif self._optimisationMode == 1:
            outcome = self._backtrackRotationOptimisation(startingBoard, board, outputMatrix, pieces, 0, 0)

        if outcome:
            # Construct matrix with colours.
            self._solution = [[(0.0, 0.0, 0.0) for _ in range(boardPiece.columns())] for _ in range(boardPiece.rows())]

            print("Solution in the piece form: ")
            prettyPrintGrid(outputMatrix)
            for i in range(boardPiece.rows()):
                for j in range(boardPiece.columns()):
                    indexPiece = outputMatrix[i][j]
                    self._solution[i][j] = piecesColour[indexPiece - 1]
        return outcome

    def _backtrackNoOptimisation(self, currentBoard: Board, board: Board, outputMatrix: Board,
                                 pieces: Pieces, currRow: int, currCol: int):
        # Get the first 0 in the grid as updated position.
        row, col = nextPos(currentBoard, currRow, currCol)
        if len(pieces) == 0 and row == -1 and col == -1:  # A.k.a. outside the grid and no pieces left.
            if self._logger:
                print("Puzzle completed successfully.")
                self._endTime = timer()
                print(f"Exiting Solver class: {self._endTime - self._startTime}...")
                print("---")
                print("----------------------------")
                print("---")
            prettyPrintGrid(outputMatrix)
            return True

        # Try all possible pieces (with different rotations as well).
        for piece in pieces:
            # Move to next position with the remaining pieces if at least one rotation is valid.
            for _ in range(4):
                decision, newRow, newCol = isValid(currentBoard, board, self._colourMap, piece, row, col)
                # if self._logger:
                #     print(f"Trying piece {piece} in position {row, col}. \nWIth current board")
                #     prettyPrintGrid(currentBoard)
                if decision:
                    setPiece(currentBoard, board, outputMatrix, piece, newRow, newCol)
                    # if len(pieces) >= 3:
                    #     print("Heh")
                    #     prettyPrintGrid(outputMatrix)
                    # Remove from list of pieces.
                    pieces.remove(piece)
                    if self._backtrackNoOptimisation(currentBoard, board, outputMatrix, pieces, newRow, newCol):
                        return True
                    pieces.append(piece)

                    # Backtrack, remove the piece.
                    removePiece(currentBoard, piece, newRow, newCol)
                rotatePieceNonOptimal(piece)
        return False

    def _backtrackRotationOptimisation(self, currentBoard: Board, board: Board, outputMatrix: Board,
                                       pieces: Pieces, currRow: int, currCol: int):
        # Get the first 0 in the grid as updated position.
        row, col = nextPos(currentBoard, currRow, currCol)
        if len(pieces) == 0 and row == -1 and col == -1:  # A.k.a. outside the grid and no pieces left.
            if self._logger:
                print("Puzzle completed successfully.")
                self._endTime = timer()
                print(f"Exiting Solver class: {self._endTime - self._startTime}...")
                print("---")
                print("----------------------------")
                print("---")
            prettyPrintGrid(outputMatrix)
            return True

        # Try all possible pieces (with different rotations as well).
        for piece in pieces:
            # Move to next position with the remaining pieces if at least one rotation is valid.
            rgLen = self._pickRangeOptimiser(piece)
            for _ in range(rgLen):
                decision, newRow, newCol = isValid(currentBoard, board, self._colourMap, piece, row, col)
                # if self._logger:
                #     print(f"Trying piece {piece} in position {row, col}. \nWIth current board")
                #     prettyPrintGrid(currentBoard)
                if decision:
                    setPiece(currentBoard, board, outputMatrix, piece, newRow, newCol)
                    # if len(pieces) >= 3:
                    #     print("Heh")
                    #     prettyPrintGrid(outputMatrix)
                    # Remove from list of pieces.
                    pieces.remove(piece)
                    if self._backtrackNoOptimisation(currentBoard, board, outputMatrix, pieces, newRow, newCol):
                        return True
                    pieces.append(piece)

                    # Backtrack, remove the piece.
                    removePiece(currentBoard, piece, newRow, newCol)
                rotatePiece(piece)
        return False

    def _pickRangeOptimiser(self, piece):
        if self._optimisationMode == 0:
            return 4
        return piece.getRotations()

    def _extractColourMap(self, board: Piece):
        self._colourMap = [[(0.0, 0.0, 0.0) for _ in range(board.columns())] for _ in range(board.rows())]
        unit = board.getUnitLen()
        (topLeftX, topLeftY) = board.getTopLeft()

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
                    b, g, r = self._image[int(centreY)][int(centreX)]
                    self._colourMap[i][j] = (r, g, b)

    def getSolution(self):
        return self._solution
