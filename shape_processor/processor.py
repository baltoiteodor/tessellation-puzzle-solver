import numpy as np
import cv2 as cv
MAXREZ = 1024
DECREMENT = 0.1


class Processor():
    def __init__(self, contours): 
        self.contours = contours
        self.pieces = []
        self.totalArea = 0
    def findUnit(self):
        # Declare a dictionary from contours to their minimum rectangle for future use
        # Find the smallest edge first, this will be a potential candidate for the unit length.
        smallestEdge = MAXREZ
        for c in self.contours:
            print("Piece: ", c.squeeze())
            self.totalArea += cv.contourArea(c)
            # min_rect = cv.minAreaRect(c)
            # contourToRectangle[c] = min_rect
            vertices = c.squeeze()
            # Calculate distances between consecutive vertices
            distances = [np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
            # Find the minimum distance
            smallestEdgeContour = min(distances)
            smallestEdge = min(smallestEdgeContour, smallestEdge)
        print("Smallest edge - ", smallestEdge)
        # Find a length such that the error is less than 5% for now. 
        unitLen = smallestEdge
        error = 1.0

        # Testing purposes
        output_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        color_contour = (0, 255, 0)  # Green color for contours
        color_rect = (0, 0, 255)  # Red color for rectangles

        while error > 0.05 and unitLen > 0:
            # Make grid for each contour
            self.pieces = []
            error = 0.0
            for c in self.contours:
                coveredArea = 0.0
                pieceArea = cv.contourArea(c)
                rect = cv.minAreaRect(c)
                centre, size, angle = rect

                cv.drawContours(output_image, [c], -1, color_contour, 2)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(output_image, [box], 0, color_rect, 2)

                # I cannot figure out why the dimensions keep on switching randomly. TopLeft will be min x min y.

                topLeftX = np.min(box[:,0])
                topLeftY = np.min(box[:,1])
                botRightX = np.max(box[:,0])
                botRightY = np.max(box[:,1])

                # For each unit of the grid, check if the centre is inside the polygon, if yes, then put 1 inside the
                # grid, otherwise 0.
                # Start with row 0, stop when we are outside the rectangle. Same for columns. 
                unitX = topLeftX
                indexX = 0
                # print("Is same width?: ", width, botRightX-topLeftX)
                # Due to width/height switching I will calculate my own.
                width = botRightX - topLeftX
                height = botRightY - topLeftY
                grid = np.zeros((int(width / unitLen + 1), int(height / unitLen + 1)))
                # print(grid)
                while unitX < botRightX: # When the new unit x coordinate is out of bounds.
                    indexY = 0
                    unitY = topLeftY
                    # Loop columns.
                    while unitY < botRightY:
                        # Find centre of grid unit, check if inside the contour.
                        centreUnit = (int(unitX + unitLen / 2), int(unitY + unitLen / 2)) 
                        isIn = cv.pointPolygonTest(c, centreUnit, False)
                        
                        if isIn >= 0: 
                            # Mark this unit as 1 in the grid. 
                            grid[indexX][indexY] = 1
                        else: 
                            # Mark as 0.     
                            grid[indexX][indexY] = 0
                        # Add to covered area
                        coveredArea += grid[indexX][indexY] * unitLen * unitLen
                        unitY += unitLen
                        indexY += 1
                    unitX += unitLen
                    indexX += 1
                # They appear as 1.0 for some reason.
                grid = grid.astype(int)
                # Remove the last column if all zero.
                if np.all(grid[:, -1] == 0):
                    grid = grid[:, :-1]

                # Remove the last row if all zero.
                if np.all(grid[-1, :] == 0):
                    grid = grid[:-1, :]

                self.pieces.append(grid)
                # Error is the maximum error per piece.
                error = max(error, (abs(1 - coveredArea / pieceArea)))
                # TODO add here a break point lol

            cv.imwrite('cnt_and_rect.jpg', output_image)


            # Current error is calculated from the ratio between total area and covered area. 
            # error = abs(1 - (coveredArea / self.totalArea))
            # print(unitLen, error)
            unitLen -= DECREMENT
        
    def getPieces(self):
        return self.pieces
    
