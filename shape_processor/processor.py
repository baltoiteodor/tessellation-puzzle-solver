import numpy as np
import cv2 as cv
from timeit import default_timer as timer

from numpy import shape

from misc.piece import Piece
from misc.types import *
from misc.contour import *

MAXREZ = 1024
DECREMENT = 0.1


# Center is in width, height format
def colourCenteredAt(image, center):
    x, y = center
    # print(center, shape(image))
    b, g, r = image[y, x]
    height, width = image.shape[:2]
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    num = 1
    for i in range(4):
        ny = y + dy[i]
        nx = x + dx[i]
        if 0 <= ny < height and 0 <= nx < width:
            bs, gs, rs = image[y + dy[i], x + dx[i]]
            b += bs
            g += gs
            r += rs
            num += 1

    return r // num, g // num, b // num


class Processor:
    def __init__(self, contours, logger: bool):
        self._contours: Contours = contours
        self._logger = logger
        self.totalArea = 0
        self._pieces = []
        self._startTime = self._endTime = 0

    def findUnit(self, image):
        # Declare a dictionary from contours to their minimum rectangle for future use
        # Find the smallest edge first, this will be a potential candidate for the unit length.

        if self._logger:
            self._startTime = timer()
            print(f"Entering Processor class...")
            print("Trying to find a good unit length for the grids of the pieces...")
            print("Looking for the smallest edge in the contours for a starting value.")

        smallestEdge = MAXREZ
        for c in self._contours:
            self.totalArea += c.getArea()
            vertices = c.getContour().squeeze()
            # Calculate distances between consecutive vertices
            distances = [np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
            # Find the minimum distance
            smallestEdgeContour = min(distances)
            smallestEdge = min(smallestEdgeContour, smallestEdge)

        if self._logger:
            print("Smallest edge - ", smallestEdge)

        # Find a length such that the error is less than 5% for now. 
        unitLen = smallestEdge + 1.0
        error = 1.0

        # Testing purposes
        # TODO: remove this sometime.
        # output_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # color_contour = (0, 255, 0)  # Green color for contours
        # color_rect = (0, 0, 255)  # Red color for rectangles

        while error > 0.05 and unitLen > 0:
            # Make grid for each contour
            self._pieces = []
            error = 0.0
            for c in self._contours:
                coveredArea = 0.0
                pieceArea = c.getArea()
                rect = c.getMinAreaRect()

                # cv.drawContours(output_image, [c.getContour()], -1, color_contour, 2)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                # cv.drawContours(output_image, [box], 0, color_rect, 2)

                # I cannot figure out why the dimensions keep on switching randomly. TopLeft will be min x min y.

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
                        isIn = cv.pointPolygonTest(c.getContour(), centreUnit, False)

                        if isIn >= 0:
                            # Mark this unit as 1 in the grid. 
                            grid[indexY][indexX] = 1
                            # originalCoord = c.getOriginalCoord(centreUnit)
                            # colours[indexY][indexX] = colourCenteredAt(image, originalCoord)
                            noOnes += 1
                        else:
                            # Mark as 0.     
                            grid[indexY][indexX] = 0
                            # colours[indexY][indexX] = (0, 0, 0)
                        # Add to covered area
                        coveredArea += grid[indexY][indexX] * unitLen * unitLen
                        unitY += unitLen
                        indexY += 1
                    unitX += unitLen
                    indexX += 1
                # They appear as 1.0 for some reason.
                grid = grid.astype(int)
                # colours = np.array(colours)
                # Remove the last columns if all zero.
                while np.all(grid[:, -1] == 0):
                    grid = grid[:, :-1]
                    # colours = colours[:, :-1]
                # Remove leading columns with all zeros
                while np.all(grid[:, 0] == 0):
                    grid = grid[:, 1:]
                    # colours = colours[:, 1:]

                # Remove the last row if all zero.
                while np.all(grid[-1, :] == 0):
                    grid = grid[:-1, :]
                    # colours = colours[:-1, :]

                # Remove leading rows with all zeros
                while np.all(grid[0, :] == 0):
                    grid = grid[1:, :]
                    # colours = colours[1:, :]

                newPiece: Piece = Piece(c, grid, c.getColour(), unitLen, (topLeftX, topLeftY))
                newPiece.canBeBoard(noOnes == newPiece.area())
                self._pieces.append(newPiece)
                # Error is the maximum error per piece.
                error = max(error, (abs(1 - coveredArea / pieceArea)))

                if error > 0.05:
                    # No point in trying for other pieces.
                    break

            # cv.imwrite('cnt_and_rect.jpg', output_image)
            unitLen -= DECREMENT
        # While loop complete, should have the pieces ready.
        if self._logger:
            self._endTime = timer()
            print("Grids completed successfully.")
            print(f"Exiting Processor class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")

    def getPieces(self):
        return self._pieces
