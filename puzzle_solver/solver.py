from puzzle_solver.helper import *
from timeit import default_timer as timer


class Solver:
    def __init__(self, logger: bool):
        self._logger = logger
        self._startTime = self._endTime = 0

    # The solve method will take as input an array of 2d arrays representing puzzle pieces and try to solve the puzzle.
    def solveBackTracking(self, pieces: Pieces):

        # We will use a backtracking algo where we try to fit a piece of the puzzle in the first empty unit
        # found. # TODO:  We try all 4 rotations of that piece if not rectangular (piece.py is rect if there are no 0
        # in its matrix representation).

        if self._logger:
            self._startTime = timer()
            print("Entering Solver class...")
            print("Attempting to solve the puzzle using basic backtracking algorithm.")

        boardPiece = findBoard(pieces)
        board = boardPiece.getGrid()

        if self._logger:
            print(f"Puzzle Board:")
            print(board)

        # Add an order number such that we can differentiate pieces in the output matrix.
        # piecesIndex = [(x, index + 1) for index, x in enumerate(pieces)]
        for index, piece in enumerate(pieces):
            piece.setOrderNumber(index + 1)

        # We will start the bkt from empty board.
        startingBoard = emptyBoard(boardPiece.rows(), boardPiece.columns())
        outputMatrix = emptyBoard(boardPiece.rows(), boardPiece.columns())

        if self._logger:
            print("Indexed pieces: ")
            print(pieces)

        return self._backtrack(startingBoard, board, outputMatrix, pieces, 0, 0)

    def _backtrack(self, currentBoard: Board, board: Board, outputMatrix: Board,
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
            rgLen = 4
            if not piece.isRotatable():
                rgLen = 1

            for _ in range(rgLen):
                decision, newRow, newCol = isValid(currentBoard, board, piece, row, col)
                if decision:
                    # print("After valid: ", newRow, newCol)
                    setPiece(currentBoard, board, outputMatrix, piece, newRow, newCol)
                    # Remove from list of pieces.
                    pieces.remove(piece)
                    if self._backtrack(currentBoard, board, outputMatrix, pieces, newRow, newCol):
                        return True
                    pieces.append(piece)

                    # Backtrack, remove the piece.
                    removePiece(currentBoard, piece, newRow, newCol)
                rotatePiece(piece)
        return False
