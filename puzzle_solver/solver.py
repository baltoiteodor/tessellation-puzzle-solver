import numpy as np


# Sorting function for the sorted method. Biggest product between the 2 sizes first.
def sortFunction(list):
    return len(list) * len(list[0])


def removePiece(currBoard, piece, row, col):
    for i in range(row, row + len(piece)):
        for j in range(col, col + len(piece[0])):
            currBoard[i][j] -= piece[i - row][j - col]

def setPiece(currBoard, board, piece, row, col):
    for i in range(row, row + len(piece)):
        for j in range(col, col + len(piece[0])):
            currBoard[i][j] = board[i][j]


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
        # We will start the bkt from empty board.
        emptyBoard = np.zeros_like(board)
        return self.backtrack(emptyBoard, board, sortedPieces, 0, 0)

    # Returns True if the piece fits in nicely, otherwise False.
    def isValid(self, currBoard, board, piece, row, col):
        print(row, col)
        print(len(currBoard), len(currBoard[0]))
        print(len(piece), len(piece[0]))
        if row + len(piece) > len(currBoard) or col + len(piece[0]) > len(currBoard[0]):
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
        return 0, 0

    def backtrack(self, currentBoard, board, pieces, currRow, currCol):
        if len(pieces) == 0:  # A.k.a. outside the grid.
            return True

        # Get the first 0 in the grid as updated position.
        row, col = self.nextPos(currentBoard, currRow, currCol)

        # Try all possible pieces (with different rotations as well?).
        for piece in pieces:
            # Move to next position with the remaining pieces.
            if self.isValid(currentBoard, board, piece, row, col):
                setPiece(currentBoard, board, piece, row, col)
                remainingPieces = pieces.copy()
                print(remainingPieces)
                print(piece)
                # Pieces is an inconvenient list of np arrays.
                remainingPieces = [arr for arr in remainingPieces if not np.array_equal(arr, piece)]
                print(remainingPieces)
                if self.backtrack(currentBoard, board, remainingPieces, row, col):
                    return True
                # Backtrack, remove the piece.
                removePiece(currentBoard, piece, row, col)


        return False
