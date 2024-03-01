import numpy as np


# Sorting function for the sorted method. Biggest product between the 2 sizes first.
# def sortFunction(list):
#     return len(list) * len(list[0])

def findBoard(pieces):
    board = None
    maxSize = 0
    boardIndex = -1

    for index, piece in enumerate(pieces):
        if np.all(piece == 1):
            currentSize = len(piece) * len(piece[0])

            if(currentSize > maxSize):
                maxSize = currentSize
                board = piece
                boardIndex = index

    if maxSize > 0:
        del pieces[boardIndex]
    return board
def removePiece(currBoard, piece, row, col):
    # cnt0 = 0
    # while piece[0][cnt0] == 0:
    #     cnt0 += 1
    #
    # # Subtract from current position, from the column cnt0.
    # if col >= cnt0:
    #     col -= cnt0
    for i in range(row, row + len(piece)):
        for j in range(col, col + len(piece[0])):
            currBoard[i][j] -= piece[i - row][j - col]


def setPiece(currBoard, board, outputMatrix, piece, row, col):
    # cnt0 = 0
    # while piece[0][0][cnt0] == 0:
    #     cnt0 += 1
    #
    # # Subtract from current position, from the column cnt0.
    # if col >= cnt0:
    #     col -= cnt0

    for i in range(row, row + len(piece[0])):
        for j in range(col, col + len(piece[0][0])):
            currBoard[i][j] += piece[0][i - row][j - col]
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

        # We will use a backtracking algo where we try to fit a piece of the puzzle in the first empty unit
        # found. We try all 4 rotations of that piece if not rectangular (Piece is rect if there are no 0
        # in its matrix representation).

        board = findBoard(pieces)

        # Add an index such that we can differentiate pieces in the output matrix.
        piecesIndex = [(x, index + 1) for index, x in enumerate(pieces)]

        # We will start the bkt from empty board.
        emptyBoard = np.zeros_like(board)
        outputMatrix = np.zeros_like(board)
        print("Indexed pieces: ")
        print(piecesIndex)
        return self.backtrack(emptyBoard, board, outputMatrix, piecesIndex, 0, 0)

    # Returns True if the piece fits in nicely, otherwise False.
    def isValid(self, currBoard, board, piece, row, col):
        # Pieces will have leading 0s in the matrix like the + sign. In this case, change the row, piece of where to put
        # the piece by the leading amount of 0s on the first row. (I think)
        cnt0 = 0
        while piece[0][cnt0] == 0:
            cnt0 += 1

        # Subtract from current position, from the column cnt0.
        if col >= cnt0:
            col -= cnt0
        # if cnt0:
        #     print("THIS SHIT HAPPENED.", cnt0)
        if row + len(piece) - 1 > len(currBoard) or col + len(piece[0]) - 1 > len(currBoard[0]):
            return False
        if cnt0:
            print("AM I GETTING HERE")
        for i in range(row, row + len(piece)):
            for j in range(col, col + len(piece[0])):
                if currBoard[i][j] + piece[i - row][j - col] > board[i][j]:
                    return False
        if cnt0:
            print("AM I GETTING HERE THO?")
        return True

    def nextPos(self, currBoard, row, col):
        print(currBoard)
        print("row col: ", row, col)
        for i in range(len(currBoard)):
            for j in range(len(currBoard[0])):
                if currBoard[i][j] == 0:
                    print("do you get here blud.")
                    return i, j
        return -1, -1

    def backtrack(self, currentBoard, board, outputMatrix, pieces, currRow, currCol):
        # Get the first 0 in the grid as updated position.
        row, col = self.nextPos(currentBoard, currRow, currCol)
        print("WTH: ", row, col)
        print("Pieces: ", pieces)
        if len(pieces) == 0 and row == -1 and col == -1:  # A.k.a. outside the grid and no pieces left.
            # print(outputMatrix)
            return True

        # Try all possible pieces (with different rotations as well).
        for piece in pieces:
            # Move to next position with the remaining pieces if at least one rotation is valid.
            # TODO: add some extra check here such that we only rotate pieces that make sense. a.i. Not squares.
            curr_piece = piece[0]
            # if piece[1] == 1:
            #     print("Current piece does it fit in any way?: ")
            # print(curr_piece)
            for _ in range(4):
                if self.isValid(currentBoard, board, curr_piece, row, col):
                    setPiece(currentBoard, board, outputMatrix, (curr_piece, piece[1]), row, col)
                    # if piece[1] == 1:
                    #     print("It isss. Current state of board")
                    #     print(currentBoard)
                    #     print("new row and col: ", row, col)
                    remainingPieces = pieces.copy()
                    # Pieces is an inconvenient list of np arrays.
                    remainingPieces = [arr for arr in remainingPieces if not (np.array_equal(arr[0], piece[0]) and arr[1] == piece[1])]
                    if self.backtrack(currentBoard, board, outputMatrix, remainingPieces, row, col):
                        return True
                    # Backtrack, remove the piece.
                    removePiece(currentBoard, curr_piece, row, col)
                curr_piece = rotatePiece(curr_piece)
        return False
