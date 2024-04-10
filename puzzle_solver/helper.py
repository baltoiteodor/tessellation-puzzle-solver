import cv2
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000

from misc.piece import *


def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)

COLOURTHRESHOLD = 25


def findBoard(pieces: Pieces):
    board = None
    maxSize = 0
    boardIndex = -1

    for index, piece in enumerate(pieces):
        # Check only pieces that are not rotatable (a.k.a. pieces containing only 1s).
        if piece.isBoardable():
            currentSize = piece.area()

            if currentSize > maxSize:
                maxSize = currentSize
                board = piece
                boardIndex = index

    if maxSize > 0:
        del pieces[boardIndex]
    return board


def removePiece(currBoard: Board, piece: Piece, row: int, col: int):
    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            currBoard[i][j] -= piece.pixelAt(i - row, j - col)


def setPiece(currBoard: Board, board: Board, outputMatrix: Board,
             piece: Piece, row: int, col: int):
    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            if currBoard[i][j] == 0:
                outputMatrix[i][j] = piece.orderNum()
            currBoard[i][j] += piece.pixelAt(i - row, j - col)


def rotatePiece(piece: Piece):
    piece.rotatePiece()

def rotatePieceNonOptimal(piece: Piece):
    rotatedGrid = np.zeros((piece.columns(), piece.rows()), dtype=int)
    # print(rotatedGrid)
    for i in range(piece.rows()):
        for j in range(piece.columns()):
            rotatedGrid[j][piece.rows() - i - 1] = piece.pixelAt(i, j)

    piece.setGrid(rotatedGrid)
    oldRows = piece.rows()
    oldCols = piece.columns()
    piece.setRowsNum(oldCols)
    piece.setColsNum(oldRows)
    piece.increaseCurrentRotation()


# Returns True if the piece fits in nicely, otherwise False.
def isValid(currBoard: Board, targetBoard: Board, colourMap, piece: Piece, row: int, col: int, colourMatters: bool):
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
            if piece.pixelAt(i - row, j - col) != 0 and \
                    (colourMatters and not similarColours(piece.getColour(), colourMap[i][j])):
                return False, row, col
            if currBoard[i][j] + piece.pixelAt(i - row, j - col) > targetBoard[i][j]:
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


def similarColours(colour1, colour2):
    lab1 = convert_color(sRGBColor(colour1[0], colour1[1], colour1[2]), LabColor)
    lab2 = convert_color(sRGBColor(colour2[0], colour2[1], colour2[2]), LabColor)

    distance = delta_e_cie2000(lab1, lab2)
    # print("Colours + distance: ", colour1, colour2, distance)
    return distance < COLOURTHRESHOLD
    # difference = np.sqrt(np.sum((np.array(colour1) - np.array(colour2))**2))
    # return difference < COLOURTHRESHOLD
