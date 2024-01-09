import numpy as np
import cv2 as cv
MAXREZ = 1024


class Processor():
    def __init__(self, contours): 
        self.contours = contours
        self.pieces = []
        self.totalArea = 0
    def findUnit(self):
        # Declare a dictionary from contours to their minimum rectangle for future use
        contourToRectangle = dict()
        # Find the smallest edge first, this will be a potential candidate for the unit length.
        smallestEdge = MAXREZ
        for c in self.contours:
            self.totalArea += cv.contourArea(c)
            # min_rect = cv.minAreaRect(c)
            # contourToRectangle[c] = min_rect
            vertices = c.squeeze()
            # Calculate distances between consecutive vertices
            distances = [np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
            # Find the minimum distance
            smallestEdgeContour = min(distances)
            smallestEdge = min(smallestEdgeContour, smallestEdge)
        
        # Find a length such that the error is less than 5% for now. 
        unitLen = smallestEdge
        error = 1.0
        while error > 0.05 and unitLen > 0:
            # Make grid for each contour
            coveredArea = 0.0
            self.pieces = []
            for c in self.contours:
                centre, size, angle = cv.minAreaRect(c)
                width, height = size
                topLeftX = int(centre[0] - width / 2)
                topLeftY = int(centre[1] - height / 2)
                print(topLeftX, topLeftY)
                # For each unit of the grid, check if the centre is inside the polygon, if yes, then put 1 inside the grid, otherwise 0.
                # Start with row 0, stop when we are outside the rectangle. Same for columns. 
                unitX = topLeftX
                indexX = 0
                grid = np.zeros((int(width / unitLen + 1), int(height / unitLen + 1)))
                print(grid)
                while unitX < int(centre[0] + width / 2): # When the new unit x coordinate is out of bounds.
                    indexY = 0
                    unitY = topLeftY
                    # Loop columns.
                    while unitY < int(centre[1] + height / 2):
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
                print(grid)
                self.pieces.append(grid)
            # Current error is calculated from the ratio between total area and covered area. 
            error = abs(1 - (coveredArea / self.totalArea)) 
            print(unitLen, error)
            unitLen -= 1
        
    def getPieces(self):
        return self.pieces
    
