import numpy as np

from misc.piece import *


def findBoard(pieces: Pieces):
    board = None
    maxSize = 0
    boardIndex = -1

    for index, piece in enumerate(pieces):
        # Check only pieces that are not rotatable (a.k.a. pieces containing only 1s).
        if not piece.isRotatable():
            currentSize = piece.area()

            if currentSize > maxSize:
                maxSize = currentSize
                board = piece
                boardIndex = index

    if maxSize > 0:
        del pieces[boardIndex]
    return board


def removePiece(currBoard: Board, piece: Piece, row: int, col: int):
    # cnt0 = 0
    # while piece[0][cnt0] == 0:
    #     cnt0 += 1
    #
    # # Subtract from current position, from the column cnt0.
    # if col >= cnt0:
    #     col -= cnt0
    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            currBoard[i][j] -= piece.pixelAt(i - row, j - col)


def setPiece(currBoard: Board, board: Board, outputMatrix: Board,
             piece: Piece, row: int, col: int):
    # cnt0 = 0
    # while piece[0][0][cnt0] == 0:
    #     cnt0 += 1
    #
    # # Subtract from current position, from the column cnt0.
    # if col >= cnt0:
    #     col -= cnt0

    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            if i >= len(board) or j >= len(board[0]):
                print("Naa")
                print(piece)
                print(row, col)
                prettyPrintGrid(currBoard)
            currBoard[i][j] += piece.pixelAt(i - row, j - col)
            outputMatrix[i][j] = piece.orderNum()


def rotatePiece(piece: Piece):
    # rotatedGrid: Grid = [[0] * piece.rows() for _ in range(piece.columns())]
    rotatedGrid = np.zeros((piece.columns(), piece.rows()), dtype=int)
    for i in range(piece.rows()):
        for j in range(piece.columns()):
            rotatedGrid[j][piece.rows() - i - 1] = piece.pixelAt(i, j)

    piece.setGrid(rotatedGrid)
    oldRows = piece.rows()
    oldCols = piece.columns()
    piece.setRowsNum(oldCols)
    piece.setColsNum(oldRows)


# Returns True if the piece fits in nicely, otherwise False.
def isValid(currBoard: Board, board: Board, piece: Piece, row: int, col: int):
    # Pieces will have leading 0s in the matrix like the + sign. In this case, change the row, piece of where to put
    # the piece by the leading amount of 0s on the first row. (I think)
    cnt0: int = 0
    while piece.pixelAt(0, cnt0) == 0:
        cnt0 += 1

    # Subtract from current position, from the column cnt0.
    # print("Before: ", row, col)
    if col >= cnt0:
        col -= cnt0
    # print("After: ", row, col, cnt0)
    if row + piece.rows() - 1 >= len(currBoard) or col + piece.columns() - 1 >= len(currBoard[0]):
        return False, row, col
    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            if currBoard[i][j] + piece.pixelAt(i - row, j - col) > board[i][j]:
                return False, row, col
    return True, row, col


def nextPos(currBoard: Board, row: int, col: int):
    for i in range(len(currBoard)):
        for j in range(len(currBoard[0])):
            if currBoard[i][j] == 0:
                return i, j
    return -1, -1


def emptyBoard(rows: int, cols: int):
    return [[0 for _ in range(cols)] for _ in range(rows)]
