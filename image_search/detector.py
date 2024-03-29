import cv2 as cv

PERCENT = 0.04  # Normally somewhere between 1-5%.


# Detector class will receive a contour print the vertices
class Detector:
    def __init__(self):
        pass

        # c is a contour - list of vertices.

    def detect(self, c):
        # c is the countour of our shape.
        # Countour approximation is done using approxPolyDP.
        shape = "nothing"
        perimeter = cv.arcLength(c, True)  # Perimeter of the contour.
        approx = cv.approxPolyDP(c, PERCENT * perimeter, True)  # Approxx will be a contour with smooth surfaces.
        print(approx)
