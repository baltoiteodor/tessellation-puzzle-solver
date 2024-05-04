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


def scalePiece(piece: Piece, scaleFactor, image):
    originalContour = piece.getOriginalContour().getContour()
    # print("LL", piece.getOriginalContour().getArea())

    originalContournp = np.array(originalContour, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(originalContournp)

    # Resize the bounding box dimensions
    scaledW = int(w * scaleFactor)
    scaledH = int(h * scaleFactor)

    # Scale and translate the contour
    scaledContour = ((originalContournp - (x, y)) * scaleFactor).astype(np.int32)

    # Create a mask for the contour
    mask = np.zeros_like(image[:,:,0])
    cv2.drawContours(mask, [scaledContour], -1, (255), thickness=cv2.FILLED)

    # Extract the region of interest (ROI) from the original image using the bounding box
    roi = image[y:y+h, x:x+w]

    # Resize the ROI using the scaled dimensions
    scaledRoi = cv2.resize(roi, (scaledW, scaledH), interpolation=cv2.INTER_LINEAR)

    # Create a mask for the scaled contour
    scaledMask = np.zeros_like(scaledRoi[:,:,0])
    cv2.drawContours(scaledMask, [scaledContour], -1, (255), thickness=cv2.FILLED)

    # Apply the scaled mask to the scaled ROI
    scaledRoiWithMask = cv2.bitwise_and(scaledRoi, scaledRoi, mask=scaledMask)

    # Save the scaled ROI with mask as an image
    cv2.imwrite('scaled.png', scaledRoiWithMask)

    # Transform into Contour with new image.
    newContour = Contour(scaledContour, scaledRoiWithMask, piece.orderNum())
    unitLen = piece.getUnitLen()

    # Create new grid. If the error is too great return failure to scale the board.
    coveredArea = 0.0
    pieceArea = newContour.getArea()

    box = cv2.boxPoints(newContour.getMinAreaRect())
    box = np.int0(box)
    # x is width, y is height.
    topLeftX = np.min(box[:, 0])
    topLeftY = np.min(box[:, 1])
    botRightX = np.max(box[:, 0])
    botRightY = np.max(box[:, 1])

    # For each unit of the grid, check if the centre is inside the polygon, if yes, then put 1 inside the
    # grid, otherwise 0.
    # Start with row 0, stop when we are outside the rectangle. Same for columns.
    unitX = topLeftX
    indexX = 0
    # print("Is same width?: ", width, botRightX-topLeftX)
    # Due to width/height switching I will calculate my own.
    width = botRightX - topLeftX
    height = botRightY - topLeftY
    rows = int(width / unitLen + 1)
    cols = int(height / unitLen + 1)
    # Invert the x, y to y, x in the grid, so it looks like in the image.
    grid = np.zeros((cols, rows))
    # colours = [[(0.0, 0.0, 0.0) for _ in range(rows)] for _ in range(cols)]
    # Use this to determine if the piece is rotatable.
    noOnes: int = 0
    while unitX < botRightX:  # When the new unit x coordinate is out of bounds.
        indexY = 0
        unitY = topLeftY
        # Loop columns.
        while unitY < botRightY:
            # Find centre of grid unit, check if inside the contour.
            centreUnit = (int(unitX + unitLen / 2), int(unitY + unitLen / 2))
            isIn = cv2.pointPolygonTest(newContour.getContour(), centreUnit, False)

            if isIn >= 0:
                # Mark this unit as 1 in the grid.
                grid[indexY][indexX] = 1
                noOnes += 1
            else:
                grid[indexY][indexX] = 0
            # Add to covered area
            coveredArea += grid[indexY][indexX] * unitLen * unitLen
            unitY += unitLen
            indexY += 1
        unitX += unitLen
        indexX += 1

    grid = grid.astype(int)
    # Remove borderline zeroes.
    grid = trimGrid(grid)

    newPiece: Piece = Piece(newContour, grid, newContour.getColour(), unitLen, (topLeftX, topLeftY))
    # Error is the maximum error per piece.
    error = abs(1 - coveredArea / pieceArea)


    if error > 0.05:
        # No point in trying for other pieces.
        return False, None
    return True, newPiece

# Rounds to closest 0.05.
def roundScaler(scaler):
    return round(scaler * 20) / 20
def calculatePiecesArea(pieces: Pieces):
    area = 0.0
    for pc in pieces:
        area += pc.getOriginalContour().getArea()

    return area
def trimGrid(grid):
    # Remove the last columns if all zero.
    while np.all(grid[:, -1] == 0):
        grid = grid[:, :-1]
    # Remove leading columns with all zeros
    while np.all(grid[:, 0] == 0):
        grid = grid[:, 1:]
    # Remove the last row if all zero.
    while np.all(grid[-1, :] == 0):
        grid = grid[:-1, :]
    # Remove leading rows with all zeros
    while np.all(grid[0, :] == 0):
        grid = grid[1:, :]

    return grid
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
    return distance < COLOURTHRESHOLD
