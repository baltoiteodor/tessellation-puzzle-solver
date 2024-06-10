import cv2 as cv
import numpy as np
from timeit import default_timer as timer

from misc.contour import Contour

thresholdAngle = 3
thresholdHeightDifference = 3


class Rotator:
    def __init__(self, logger: bool):
        self._logger = logger
        self._startTime = self._endTime = 0

    def getTimeTaken(self):
        return self._endTime - self._startTime
    def rotate(self, contours, image):
        self._startTime = timer()
        if self._logger:
            print("Entering Rotator class...")
            print("Checking shapes if they need rotating...")

        rotatedImage = np.zeros_like(image)
        # rotatedContours = []

        # cv.drawContours(rotatedImage, contours, 0, (255, 0, 0), 2)

        for i, contour in enumerate(contours):
            # Calculate the lowest point and lowest of its neighbours.
            lowestPoint, secondPoint = lowestSide(contour)
            # Calculate if the lowest edge is parallel to Ox by checking if the y coordinates are similar.
            yDifference = abs(lowestPoint[0][1] - secondPoint[0][1])
            # If it is parallel then do not do anything to the contour.
            if yDifference < thresholdHeightDifference:
                if self._logger:
                    print(f"Difference in angle not above certain threshold for contour number {i}.")
                # rotatedContours.append(Contour(contour, image, i))
                # Also draw for debugging purposes.
                if self._logger:
                    cv.drawContours(rotatedImage, [contour.getContour()], 0, (255, 255, 255), 2)
            else:
                # Need to calculate the angle and rotate the piece to be straight.
                angle = np.arctan2(lowestPoint[0][1] - secondPoint[0][1],
                                   lowestPoint[0][0] - secondPoint[0][0]) * 180 / np.pi
                rotationAngle = - angle
                # center = (float(lowestPoint[0][0]), float(lowestPoint[0][1]))
                #
                # rotationMatrix = cv.getRotationMatrix2D(center, rotationAngle, 1.0)
                #
                # rotatedContour = cv.transform(np.array([contour]), rotationMatrix)[0]
                rotatedContour = contour.rotate(lowestPoint, rotationAngle)
                if self._logger:
                    print(f"Difference above certain threshold for contour number {i}: {contour}")
                    print("Angle:", angle)
                    print("Lowest point: ", lowestPoint)
                    print("New rotated contour: ", rotatedContour)
                    cv.drawContours(rotatedImage, [rotatedContour.astype(int)], -1, (0, 255, 0), thickness=cv.FILLED)

                contour.setContour(rotatedContour)
                # rotatedContours.append(Contour(rotatedContour, image, i))

        cv.imwrite("straight.jpg", rotatedImage)
        self._endTime = timer()
        if self._logger:
            print(f"Exiting Rotator class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")

        return contours

# Given a contour, finds the side that has a point with max y coordinate. Describes it as a pair of two points.
def lowestSide(contour):
    maxY = -1
    lowestPoint = []
    secondPoint = []
    size = len(contour.getContour())
    cnt = contour.getContour()
    for i in range(size):
        if cnt[i][0][1] >= maxY:
            lowestPoint = cnt[i]
            left = []
            right = []
            if i != 0 and i != size - 1:
                left = cnt[i - 1]
                right = cnt[i + 1]
            elif i == 0:
                left = cnt[size - 1]
                right = cnt[i + 1]
            elif i == size - 1:
                left = cnt[i - 1]
                right = cnt[0]

            if left[0][1] < right[0][1]:
                secondPoint = right
            else:
                secondPoint = left
            maxY = cnt[i][0][1]
    return lowestPoint, secondPoint

