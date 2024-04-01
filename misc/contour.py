import cv2
import numpy as np

from misc.types import *


# Receives an openCV contour, construct an object that adds a colour aspect to it.
class Contour:
    def __init__(self, contour, image, ordNum: int):
        self._contour = contour
        self._image = image
        self._ordNum = ordNum
        self._initialize()
        self._colour = self._calculateColour()


    def getColour(self):
        return self._colour

    def getContour(self):
        return self._contour

    def getBoundingRect(self):
        return self._boundingRectangle

    def setContour(self, contour):
        self._contour = contour
        self._initialize()

    def getArea(self):
        return self._area

    def getMinAreaRect(self):
        return self._minAreaRect

    def _initialize(self):
        self._boundingRectangle = cv2.boundingRect(self._contour)
        self._minAreaRect = cv2.minAreaRect(self._contour)
        self._area = cv2.contourArea(self._contour)

    # Sample 11 points, return the median value of an average of surrounding
    # neighbours.
    def _calculateColour(self):
        contour = self._contour
        x, y, width, height = self._boundingRectangle
        noPoints = 11

        randomPoint = (np.random.randint(x, x + width),
                       np.random.randint(y, y + height))
        blue = []
        green = []
        red = []
        for _ in range(noPoints):
            isInside = cv2.pointPolygonTest(contour, randomPoint, measureDist=False)
            while isInside <= 0:
                randomPoint = (np.random.randint(x, x + width),
                               np.random.randint(y, y + height))
                isInside = cv2.pointPolygonTest(contour, randomPoint, measureDist=False)
            # Found a point that is inside, TODO: make up an average of neighbours.
            b, g, r = self._image[randomPoint[1], randomPoint[0]]

            blue.append(b)
            green.append(g)
            red.append(r)

        # Find the median value from the intensities of the random pixels and return it.
        medianB = np.median(blue)
        medianG = np.median(green)
        medianR = np.median(red)

        return medianR, medianG, medianB

    def __repr__(self):
        return f"Contour {self._ordNum} with colour " \
               f"{self._colour} looks like this: \n {self._contour}"


Contours = List[Contour]
