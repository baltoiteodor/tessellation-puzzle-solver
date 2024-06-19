import imutils
import cv2 as cv
import numpy as np
from timeit import default_timer as timer

from misc.contour import Contour

MinVertices = 3
MaxVertices = 50

MINASPECT = 0.09
MAXASPECT = 10.1

MAXCURVATURE = 20


class ShapeFinder:
    def __init__(self, logger: bool):
        self._logger = logger
        self._startTime = self._endTime = 0

    def detectShapes3D(self, image, originalImage):
        self._startTime = timer()
        if self._logger:
            print("Entering ShapeFinder class, function detectShapes...")

        # Uncomment to inspect incoming image.
        # cv.imwrite("imgContrastLab.png", image)

        # Calculate edges.
        edgeImage = cv.Canny(image, 10, 150)

        # Inspect output.
        # cv.imwrite("canny.png", edgeImage)

        contoursCV = cv.findContours(edgeImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contoursCV)

        contourImage = np.zeros_like(originalImage)
        contourFiltered = np.zeros_like(originalImage)

        cv.drawContours(contourImage, contours, -1, (0, 255, 0), 2)

        # Uncomment to inspect contours.
        # cv.imwrite('contoursagain.png', contourImage)

        filteredContours = []
        H, W = image.shape[:2]
        AREA = H * W

        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)

            # Area filtering:
            MINAREA = AREA / 600
            MAXAREA = AREA / 2

            if not MINAREA < area < MAXAREA:
                continue

            contour = Contour(cv.approxPolyDP(contour, 0.3, True), originalImage, i)
            filteredContours.append(contour)

        # Uncomment to inspect filtered contours.
        # cv.drawContours(contourFiltered, filteredContours, -1, (0, 255, 0), 2)
        # cv.imwrite('contoursagainButFiltered.png', contourFiltered)

        if self._logger:
            print("Contours have been found and filtered...")
            for i, contour in enumerate(contours):
                print(f"Countour {i}: ", contour)
            print("Trying to smoothen the countours...")

        # Smoothen contours - shapes at an angle might have noise.
        smoothedContours = []
        # Higher epsilon means aggressive smoothing, low epsilon keeps more of the details.
        epsilon = 0.01

        contourList = []
        for i, contour in enumerate(contours):
            # Approximate the contour to smoothen it.
            smoothContour = cv.approxPolyDP(contour, epsilon * cv.arcLength(contour, True), True)

            # Append the smoothed contour to the list if the contour has at least 4 vertices. We do not have
            # triangle pieces.
            if len(smoothContour) <= MinVertices or len(smoothContour) >= MaxVertices:
                continue

            area = cv.contourArea(smoothContour)
            MINAREA = AREA / 600
            MAXAREA = AREA / 2

            if not MINAREA < area < MAXAREA:
                continue

            smoothedContours.append(smoothContour)
            contourList.append(Contour(smoothContour, originalImage, i))

            if self._logger:
                print(f"Smoothed contour {i}: ", smoothContour)

        # Uncomment to inspect smoothed contours.
        # contourImageSmt = np.zeros_like(originalImage)
        # cv.drawContours(contourImageSmt, smoothedContours, -1, (0, 255, 0), 2)
        # cv.imwrite('contoursagainsmoothed.png', contourImageSmt)

        self._endTime = timer()
        if self._logger:
            print(f"Exiting ShapeFinder class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")

        return filteredContours

    def detectJigsaw(self, image, originalImage):
        self._startTime = timer()
        if self._logger:
            print("Entering ShapeFinder class, function detectShapes...")

        # Uncomment to inspect incoming image.
        # cv.imwrite("imgContrastLab.png", image)

        # Find the contours.
        contoursCV = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contoursCV)

        # Uncomment to inspect contours.
        # contourImage = np.zeros_like(originalImage)
        # contourFiltered = np.zeros_like(originalImage)
        # cv.drawContours(contourImage, contours, -1, (0, 255, 0), 2)
        # cv.imwrite('contoursagain.png', contourImage)

        filteredContours = []
        H, W = image.shape[:2]
        AREA = H * W
        clt = []

        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)

            # Area filtering:
            MINAREA = AREA / 600
            MAXAREA = AREA / 2

            if not MINAREA < area < MAXAREA:
                continue

            clt.append(contour)
            filteredContours.append(Contour(contour, originalImage, i))

        # Uncomment to inspect filtered contours.
        # cv.drawContours(contourFiltered, clt, -1, (0, 255, 0), 2)
        # cv.imwrite('contoursagainButFiltered.png', contourFiltered)

        if self._logger:
            print("Contours have been found and triaged...")
            for i, contour in enumerate(contours):
                print(f"Countour {i}: ", contour)

        self._endTime = timer()
        if self._logger:
            print(f"Exiting ShapeFinder class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")
        return filteredContours

    def detectShapes2D(self, image, originalImage):
        self._startTime = timer()
        if self._logger:
            print("Entering ShapeFinder class, function detectShapes...")

        # Find contours and deal with them.
        contoursP = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contoursP)

        # Uncomment to inspect contours.
        # contourImage = np.zeros_like(originalImage)
        # cv.drawContours(contourImage, contours, -1, (255, 255, 255), thickness = 1)
        # cv.imwrite('contoursagain.jpg', contourImage)


        if self._logger:
            print("Contours have been found...")
            for i, contour in enumerate(contours):
                print(f"Countour {i}: ", contour)

            print("Trying to smoothen the countours...")

        # Smoothen contours - shapes at an angle might have noise.
        smoothedContours = []
        # Higher epsilon means aggressive smoothing, low epsilon keeps more of the details.
        epsilon = 0.02

        contourList = []
        for i, contour in enumerate(contours):
            # Approximate the contour to smoothen it.
            smooth_contour = cv.approxPolyDP(contour, epsilon * cv.arcLength(contour, True), True)

            # Append the smoothed contour to the list if the contour has at least 4 vertices. There are no
            # triagle pieces.
            if len(smooth_contour) <= MinVertices or len(smooth_contour) >= MaxVertices:
                continue

            smoothedContours.append(smooth_contour)
            contourList.append(Contour(smooth_contour, originalImage, i))

            if self._logger:
                print(f"Smoothed contour {i}: ", smooth_contour)

        # Uncomment to inspect smoothed contours.
        # contourImageSmt = np.zeros_like(originalImage)
        # cv.drawContours(contourImageSmt, smoothedContours, -1, (255, 255, 255), thickness = 1)
        # cv.imwrite('contoursagainsmoothed.jpg', contourImageSmt)

        self._endTime = timer()
        if self._logger:
            print(f"Exiting ShapeFinder class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")
        return contourList

    def getTimeTaken(self):
        return self._endTime - self._startTime