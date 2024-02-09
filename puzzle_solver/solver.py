import numpy as np


# Sorting function for the sorted method. Biggest product between the 2 sizes first.
def sortFunction(list):
    return len(list) * len(list[0])


def removePiece(currBoard, piece, row, col):
    for i in range(row, row + len(piece)):
        for j in range(col, col + len(piece[0])):
            currBoard[i][j] -= piece[i - row][j - col]


def setPiece(currBoard, board, outputMatrix, piece, row, col):

    for i in range(row, row + len(piece[0])):
        for j in range(col, col + len(piece[0][0])):
            currBoard[i][j] = board[i][j]
            outputMatrix[i][j] = piece[1]

# Rotates a 2D puzzle piece, clock-wise, 90 degrees.
# TODO: add to test suite.
def rotatePiece(piece):
    numRows = len(piece)
    numCols = len(piece[0])

    rotatedPiece = [[0] * numRows for _ in range(numCols)]

    for i in range(numRows):
        for j in range(numCols):
            rotatedPiece[j][numRows - i - 1] = piece[i][j]

    return rotatedPiece
class Solver:
    def __init__(self):
        pass

    # The solve method will take as input an array of 2d arrays representing puzzle pieces and try to solve the puzzle.
    def solveBackTracking(self, pieces):
        # Start off by sorting pieces, biggest one first (it being the board).
        sortedPieces = sorted(pieces, key=sortFunction, reverse=True)

        # We will use a backtracking algo where we try to fit a piece of the puzzle in the first empty unit
        # found. We try all 4 rotations of that piece if not rectangular (Piece is rect if there are no 0
        # in its matrix representation).

        board = sortedPieces[0]
        sortedPieces = sortedPieces[1:]

        # Add an index such that we can differentiate pieces in the output matrix.
        sortedPiecesIndex = [(x, index + 1) for index, x in enumerate(sortedPieces)]
        print("What is this", sortedPiecesIndex)
        # We will start the bkt from empty board.
        emptyBoard = np.zeros_like(board)
        outputMatrix = np.zeros_like(board)

        return self.backtrack(emptyBoard, board, outputMatrix, sortedPiecesIndex, 0, 0)

    # Returns True if the piece fits in nicely, otherwise False.
    def isValid(self, currBoard, board, piece, row, col):
        if row + len(piece) - 1 > len(currBoard) or col + len(piece[0]) - 1 > len(currBoard[0]):
            return False
        for i in range(row, row + len(piece)):
            for j in range(col, col + len(piece[0])):
                if currBoard[i][j] + piece[i - row][j - col] != board[i][j]:
                    return False
        return True

    def nextPos(self, currBoard, row, col):
        for i in range(row, len(currBoard)):
            for j in range(len(currBoard[0])):
                if i >= row and j >= col and currBoard[i][j] == 0:
                    return i, j
        return -1, -1

    def backtrack(self, currentBoard, board, outputMatrix, pieces, currRow, currCol):
        # Get the first 0 in the grid as updated position.
        row, col = self.nextPos(currentBoard, currRow, currCol)

        if len(pieces) == 0 and row == -1 and col == -1:  # A.k.a. outside the grid and no pieces left.
            print(outputMatrix)
            return True

        # Try all possible pieces (with different rotations as well?).
        for piece in pieces:
            # Move to next position with the remaining pieces if at least one rotation is valid.
            # TODO: add some extra check here such that we only rotate pieces that make sense. a.i. Not squares.
            curr_piece = piece[0]
            for _ in range(4):
                if self.isValid(currentBoard, board, curr_piece, row, col):
                    setPiece(currentBoard, board, outputMatrix, (curr_piece, piece[1]), row, col)
                    remainingPieces = pieces.copy()
                    # Pieces is an inconvenient list of np arrays.
                    remainingPieces = [arr for arr in remainingPieces if not (np.array_equal(arr[0], piece[0]) and arr[1] == piece[1])]
                    if self.backtrack(currentBoard, board, outputMatrix, remainingPieces, row, col):
                        return True
                    # Backtrack, remove the piece.
                    removePiece(currentBoard, curr_piece, row, col)
                curr_piece = rotatePiece(curr_piece)
        return False
