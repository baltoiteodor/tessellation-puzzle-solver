import math

import cv2 as cv
from timeit import default_timer as timer

from misc.piece import Piece
from misc.contour import *
from puzzle_solver.helper import trimGrid, findClosestContourPoint, resizeToDimensions
import numpy as np

MAXSIZE = 1024
DECREMENT = 0.1


# Center is in width, height format. Sample points from the cell to determine the colour.
def colourCenteredAt(image, center):
    x, y = center
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


# Fill the X-frame for each grid.
def fillTwos(grid):
    r = len(grid)
    c = len(grid[0])
    ti = 0
    tj = 0
    ok = False
    for i in range(r):
        if ok:
            break
        for j in range(c):
            if grid[i][j] == 2:
                ti = i
                tj = j
                ok = True
                break
    ti += 1
    tj += 1

    # ti, tj is now the middle 2, compute the other 2 2s.

    ti -= 1
    tj += 1
    if grid[ti][tj] == 1:
        grid[ti][tj] = 2

    ti += 2
    tj -= 2
    if grid[ti][tj] == 1:
        grid[ti][tj] = 2

    return grid


# Based on a starting point, we look for the furthest point along Ox to find the right corner of the jigsaw piece.
def findFurthestPointHorizontal(contour, startPoint, yThreshold):
    furthestPoint = startPoint
    maxDistance = -1

    for point in contour:
        point = point[0]
        # Check if the point is within the yThreshold. This is to combat noise.
        if abs(point[1] - startPoint[1]) <= yThreshold:
            # Check if this point is further to the right.
            if point[0] > startPoint[0]:
                distance = point[0] - startPoint[0]
                if distance > maxDistance:
                    maxDistance = distance
                    furthestPoint = point

    return tuple(furthestPoint)


# Based on a starting point, we look for the furthest point along Oy to find
# the bottom left corner of the jigsaw piece.
def findFurthestPointVertical(contour, startPoint, xThreshold):
    furthestPoint = startPoint
    maxDistance = -1

    for point in contour:
        point = point[0]
        # Check if the point is within the xThreshold. To combat noise.
        if abs(point[0] - startPoint[0]) <= xThreshold:
            # Check if this point is further down.
            if point[1] > startPoint[1]:
                distance = point[1] - startPoint[1]
                if distance > maxDistance:
                    maxDistance = distance
                    furthestPoint = point

    return tuple(furthestPoint)


class Processor:
    def __init__(self, contours, logger: bool, jigsawMode: bool):
        self._contours: Contours = contours
        self._logger = logger
        self.totalArea = 0
        self._pieces = []
        self._startTime = self._endTime = 0
        self._jigsawMode = jigsawMode

    def findGrids(self):
        # Find the smallest edge first, this will be a potential candidate for the unit length.

        self._startTime = timer()
        if self._logger:
            print(f"Entering Processor class...")
            print("Trying to find a good unit length for the grids of the pieces...")
            print("Looking for the smallest edge in the contours for a starting value.")

        smallestEdge = self._findMinUnitLen()

        if self._logger:
            print("Smallest edge - ", smallestEdge)

        # Find a length such that the error is less than 5% for now. 
        unitLen = smallestEdge + 1.0
        error = 1.0

        while error > 0.05 and unitLen > 0:
            # Make grid for each contour.
            self._pieces = []
            error = 0.0
            for c in self._contours:
                coveredArea = 0.0
                pieceArea = c.getArea()

                x, y, w, h = c.getBoundingRect()

                # Create a rotated rectangle manually from the bounding rectangle
                center = (x + w / 2, y + h / 2)
                size = (w, h)
                angle = 0  # Since it's a straight rectangle, the angle is 0.
                rect = (center, size, angle)

                box = cv2.boxPoints(rect)
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
                # Due to width/height switching I will calculate my own.
                width = botRightX - topLeftX
                height = botRightY - topLeftY
                rows = int(width / unitLen + 1)
                cols = int(height / unitLen + 1)
                # Invert the x, y to y, x in the grid, so it looks like in the image.
                grid = np.zeros((cols, rows))

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
                if len(grid) == 0:
                    error = 1
                    break
                newPiece: Piece = Piece(c, grid, c.getColour(), unitLen, (topLeftX, topLeftY))
                newPiece.canBeBoard(noOnes == newPiece.area(), self._jigsawMode)
                self._pieces.append(newPiece)
                # Error is the maximum error per piece.
                error = max(error, (abs(1 - coveredArea / pieceArea)))

                if error > 0.05:
                    # No point in trying for other pieces.
                    break

            unitLen -= DECREMENT
        # While loop complete, should have the pieces ready.
        self._endTime = timer()
        if self._logger:
            print("Grids completed successfully.")
            print(f"Exiting Processor class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")

    def findGridsJigsaw(self, rows, columns):
        self._startTime = timer()
        if self._logger:
            print(f"Entering Processor class...")
            print("Trying to find a good unit length for the grids of the jigsaw pieces...")

        # Retrieve the original image.
        image = self._contours[0].getImage()

        maxArea = 0
        for contour in self._contours:
            if contour.getArea() > maxArea:
                maxArea = contour.getArea()

        piecesArea = 0
        for contour in self._contours:
            if contour.getArea() != maxArea:
                piecesArea += contour.getArea()

        a = piecesArea / (9 * rows * columns)

        # Calculate unit length based on all pieces.
        unitLen = int(np.sqrt(a))

        # Uncomment to inspect the unit length.
        # print("UnitLen: ", unitLen)
        # print("Max ", maxArea)

        # Target sizes for the board.
        targetW = unitLen * 3 * columns
        targetH = unitLen * 3 * rows

        for i, contour in enumerate(self._contours):

            if contour.getArea() == maxArea:
                # Case when we are dealing with the board.
                newCont, newImg = resizeToDimensions(image, contour.getContour(), targetW, targetH)
                self._contours[i] = Contour(newCont, newImg, 0)

                # Uncomment to inspect resized contour.
                # cv.imwrite(f"debuggingScalingPieces/contour{i}.png", newImg)

                maxArea = self._contours[i].getArea()

            else:
                x, y, w, h = contour.getBoundingRect()
                topLeft = (x, y)
                leftJig = findClosestContourPoint(contour.getOriginalContour(), np.array(topLeft))
                topRight = (x + w, y)
                rightJig = findClosestContourPoint(contour.getOriginalContour(), np.array(topRight))

                # Calculate the right corner in a different manner.
                diffTryRight = findFurthestPointHorizontal(contour.getOriginalContour(), leftJig, 3)

                if diffTryRight != rightJig:
                    rightJig = diffTryRight

                pieceW = rightJig[0] - leftJig[0]
                accUnit = pieceW / 3
                scalerW = unitLen / accUnit

                botLeft = (x, y + h)

                leftBotJig = findClosestContourPoint(contour.getOriginalContour(), np.array(botLeft))
                diffTryBotJig = findFurthestPointVertical(contour.getOriginalContour(), leftJig, 3)

                if diffTryBotJig != leftBotJig:
                    leftBotJig = diffTryBotJig

                pieceH = leftBotJig[1] - leftJig[1]
                accUnit = pieceH / 3
                scalerH = unitLen / accUnit

                # Uncomment to inspect scaler for this contour.
                # print("For contour ", i)
                # print(scalerW, scalerH)

                # Calculate the target sizes of the piece and rescale.
                targetPieceW = int(w * scalerW)
                targetPieceH = int(h * scalerH)
                newCont, newImg = resizeToDimensions(image, contour.getContour(), targetPieceW, targetPieceH)

                # Uncomment to inspect the corners of each piece.
                # im = image.copy()
                # cv.circle(im, rightJig, radius=5, color=(0, 0, 255), thickness=-1)
                # cv.circle(im, diffTryRight, radius=5, color=(0, 255, 0), thickness=-1)
                # cv.circle(im, diffTryBotJig, radius=5, color=(0, 255, 0), thickness=-1)
                # cv.imwrite(f"contour{i}Jig.png", im)

                # Uncomment to inspect the rescaled image.
                # cv.imwrite(f"contour{i}.png", newImg)

                self._contours[i] = Contour(newCont, newImg, 0)

        # Will trim this grid after matching with piece.
        for i, contour in enumerate(self._contours):
            x, y, w, h = contour.getBoundingRect()
            topLeft = (x, y)
            closestPoint = findClosestContourPoint(contour.getOriginalContour(), np.array(topLeft))

            # Create an image to draw the contour and the closest point. Uncomment this for debugging if needed.
            # img = np.zeros((y+h+10, x+w+10, 3), dtype=np.uint8)
            # adjustedC = contour.getOriginalContour() - (x, y)
            # Draw the contour.
            # cv.drawContours(img, [adjustedC], -1, (0, 255, 0), 2)  # Green contour

            # Draw a circle at the closest point.
            # adjusted_closest_point = (closestPoint[0] - x, closestPoint[1] - y)
            # cv.circle(img, adjusted_closest_point, 5, (0, 0, 255), -2)
            # adjusted_closest_point = (closestPoint[0] - x - unitLen, closestPoint[1] - y - unitLen)
            # cv.circle(img, adjusted_closest_point, 5, (0, 0, 255), -2)

            ### Report purposes - Uncomment this to obtain all points detected by the Harris Corner Detector.

            # pieceImg = contour.getImage()
            # corners_image = np.zeros_like(pieceImg)
            # mask = np.zeros(pieceImg.shape[:2], dtype=np.uint8)
            # cv.drawContours(mask, [contour.getOriginalContour()], -1, 255, -1)
            # gray = cv.cvtColor(pieceImg, cv.COLOR_BGR2GRAY)
            # gray = np.float32(gray)
            # dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
            # dst = cv.dilate(dst, None)
            # threshold = 0.01 * dst.max()
            # corners_image[dst > threshold] = [0, 0, 255]
            #
            # combined_image = cv.bitwise_and(pieceImg, pieceImg, mask=mask)
            # combined_image[dst > threshold] = [0, 0, 255]
            #
            # cv.imshow('Corners', combined_image)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # Draws a point inside center of each cell.

            # for row in range(6):
            #     for col in range(6):
            #         centreX = closestPoint[0] - x - unitLen + row * unitLen
            #         centreY = closestPoint[1] - y - unitLen + col * unitLen
            #         centreUnit = (int(centreX), int(centreY))
            #         cv.circle(img, centreUnit, 5, (0, 0, 255), -2)  # Red circle

            # After obtaining the closest point, this will also be the top left corner of the jigsaw piece. Hence, we
            # create the grid based on this coordinate. We extract from this coordinate a unit length and match a 6x6
            # grid.

            gridX = closestPoint[0] - unitLen
            gridY = closestPoint[1] - unitLen
            noOnes = 0

            if contour.getArea() == maxArea:
                # Fill the board with 1s of corresponding size.
                grid = np.ones((3 * rows, 3 * columns))
                noOnes = 9 * rows * columns
                gridX += unitLen
                gridY += unitLen
            else:
                # Build the grid for the piece.
                grid = np.zeros((6, 6))
                for row in range(6):
                    for col in range(6):
                        centreX = gridX + row * unitLen + unitLen / 2
                        centreY = gridY + col * unitLen + unitLen / 2

                        # For this centre of the cell we can check if it is inside the contour and deem
                        # if we place a 1 in the grid.
                        centreUnit = (int(centreX), int(centreY))
                        isIn = cv.pointPolygonTest(contour.getContour(), centreUnit, False)
                        if isIn >= 0:
                            # Mark this unit as 1 in the grid or 2 if it is on the X.
                            grid[col][row] = 1
                            if row == col:
                                grid[col][row] = 2
                            noOnes += 1
                        else:
                            grid[col][row] = 0

            grid = grid.astype(int)
            copyGrid = grid
            # Remove borderline zeroes.
            grid = trimGrid(grid)
            grid = fillTwos(grid)

            if self._logger:
                print("For piece this is the grid: ", grid)

            newPiece: Piece = Piece(contour, grid, contour.getColour(), unitLen, (gridX, gridY))
            newPiece.canBeBoard(noOnes == newPiece.area(), self._jigsawMode)

            image = contour.getImage()
            blurredImage = cv.GaussianBlur(image, (45, 45), 0)

            # Construct the colour grid of the piece.
            newPiece.computeColourGrid(blurredImage, copyGrid)

            self._pieces.append(newPiece)

            # Show the image constructed previously if wanting to check corner detection.
            # cv.imshow('Contour with Closest Point', img)
            # cv.waitKey(0)

        self._endTime = timer()
        if self._logger:
            print("Grids completed successfully.")
            print(f"Exiting Processor class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")

    # Basically calculate a potential maximum unit length for each bounding rectangle and compare to a minimum edge over all.
    def _findMinUnitLen(self):
        smallestEdge = MAXSIZE
        for c in self._contours:
            self.totalArea += c.getArea()
            # Get the bounding rectangle, calculate its area, calculate max potential unitLen.
            _, _, w, h = c.getBoundingRect()
            area = w * h
            potentialUnitLenMax = int(math.sqrt(area))
            smallestEdge = min(smallestEdge, potentialUnitLenMax)

        return smallestEdge

    def getPieces(self):
        return self._pieces

    def getTimeTaken(self):
        return self._endTime - self._startTime
