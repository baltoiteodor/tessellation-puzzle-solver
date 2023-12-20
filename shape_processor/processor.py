import numpy as np
import cv2 as cv
MAXREZ = 1024


class Processor():
    def __init__(self, contours): 
        self.contours = contours
        self.pieces = []
    def findUnit(self):
        # Declare a dictionary from contours to their minimum rectangle for future use
        contourToRectangle = dict()
        # Find the smallest edge first, this will be a potential candidate for the unit length.
        smallestEdge = MAXREZ
        for c in self.contours:
            min_rect = cv.minAreaRect(c)
            contourToRectangle[c] = min_rect
            vertices = c.squeeze()
            # Calculate distances between consecutive vertices
            distances = [np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
            # Find the minimum distance
            smallestEdgeContour = min(distances)
            smallestEdge = min(smallestEdgeContour, smallestEdge)
        
        # Find a length such that the error is less than 5% for now. 
        unitLen = smallestEdge
        error = 1.0
        while error > 0.05:
            for c in self.contours:
                centre, size, angle = contourToRectangle[c]
                width, height = size
                topLeftX = int(centre[0] - width / 2)
                topLeftY = int(centre[1] - height / 2)

                # For each unit of the grid, check if the centre is inside the polygon, if yes, then put 1 inside the grid, otherwise 0.

    def getPieces(self):
        return self.pieces
    
