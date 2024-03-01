import cv2 as cv
import numpy as np

thresholdAngle = 3
thresholdHeightDifference = 3

# Cartesian to Polar coordinates
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

# Polar to Cartesian
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

# Given a contour, finds the side that has a point with max y coordinate. Describes it as a pair of two points.
def lowestSide(contour):
    maxY = -1
    lowestPoint = []
    secondPoint = []
    size = len(contour)
    for i in range(size):
        if contour[i][0][1] >= maxY:
            lowestPoint = contour[i]
            left = []
            right = []
            if i != 0 and i != size - 1:
                left = contour[i - 1]
                right = contour[i + 1]
            elif i == 0:
                left = contour[size - 1]
                right = contour[i + 1]
            elif i == size - 1:
                left = contour[i - 1]
                right = contour[0]

            if left[0][1] < right[0][1]:
                secondPoint = right
            else:
                secondPoint = left
            maxY = contour[i][0][1]
    return (lowestPoint, secondPoint)

class Rotator():
    def __init__(self):
        pass
    def rotate(self, contours, image):
        print("Entered rotator class.")
        rotatedImage = np.zeros_like(image)
        rotatedContours = []

        cv.drawContours(rotatedImage, contours, 0, (255, 0, 0), 2)

        for contour in contours:
            print("Current contour: ")
            print(contour)
            # Calculate the lowest point and lowest of its neighbours.
            lowestPoint, secondPoint = lowestSide(contour)
            # Calculate if the lowest edge is parallel to Ox by checking if the y coordinates are similar.
            yDifference = abs(lowestPoint[0][1] - secondPoint[0][1])
            print("After looking for lowest side here are the results: ", yDifference, lowestPoint, secondPoint)
            # If it is parallel then do not do anything to the contour.
            if yDifference < thresholdHeightDifference:
                print("Difference not above certain threshold.")
                rotatedContours.append(contour)
                # Also draw for debugging purposes.
                cv.drawContours(rotatedImage, [contour], 0, (255, 255, 255), 2)
            else:
                print("Difference above certain threshold.")
                # Need to calculate the angle and rotate the piece to be straight.
                angle = np.arctan2(lowestPoint[0][1] - secondPoint[0][1],
                                   lowestPoint[0][0] - secondPoint[0][0]) * 180 / np.pi
                print("Angle:", angle)
                rotationAngle = - angle
                print("LOWP: ", lowestPoint)
                center = (float(lowestPoint[0][0]), float(lowestPoint[0][1]))
                print("Center:", center)

                rotationMatrix = cv.getRotationMatrix2D(center, rotationAngle, 1.0)

                rotatedContour = cv.transform(np.array([contour]), rotationMatrix)[0]

                cv.drawContours(rotatedImage, [rotatedContour.astype(int)], -1, (0, 255, 0), thickness=cv.FILLED)

                rotatedContours.append(rotatedContour)

        cv.imwrite("straight.jpg", rotatedImage)
        return rotatedContours
