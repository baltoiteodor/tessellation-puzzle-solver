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
        pass

    def detectShapes3D(self, image, originalImage):
        if self._logger:
            self._startTime = timer()
            print("Entering ShapeFinder class, function detectShapes...")

        cv.imwrite("imgContrastLab.png", image)
        edgeImage = cv.Canny(image, 10, 150)
        cv.imwrite("canny.png", edgeImage)

        contoursCV = cv.findContours(edgeImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contoursCV)

        # what is going on?
        contour_image = np.zeros_like(originalImage)
        contourFiltered = np.zeros_like(originalImage)
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagain.png', contour_image)
        filteredContours = []
        H, W = image.shape[:2]
        AREA = H * W
        for contour in contours:
            area = cv.contourArea(contour)

            # Area filtering:
            MINAREA = AREA / 600
            MAXAREA = AREA / 2

            if not MINAREA < area < MAXAREA:
                continue

            # Aspect ratio (1:10 minimum):
            # x, y, w, h = cv.boundingRect(contour)
            # aspectRatio = float(w) / h
            # if not MINASPECT < aspectRatio < MAXASPECT:
            #     continue

            # Curvature of contours:
            # hull = cv.convexHull(contour)
            # defects = cv.convexityDefects(contour, cv.convexHull(contour, returnPoints=False))
            #
            # # Filter contours based on convexity defects or concave regions
            # if defects is not None:
            #     num_defects = defects.shape[0]
            #     if num_defects > MAXCURVATURE:
            #         continue

            filteredContours.append(contour)

        cv.drawContours(contourFiltered, filteredContours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagainButFiltered.png', contourFiltered)

        if self._logger:
            print("Contours have been found and triaged...")
            for i, contour in enumerate(contours):
                print(f"Countour {i}: ", contour)

            print("Trying to smoothen the countours...")
        # Smoothen contours - shapes at an angle might have noise.
        smoothed_contours = []
        # Higher epsilon means aggressive smoothing, low epsilon keeps more of the details.
        epsilon = 0.01

        contourList = []
        for i, contour in enumerate(contours):
            # Approximate the contour to smoothen it
            print("smoothing")
            smooth_contour = cv.approxPolyDP(contour, epsilon * cv.arcLength(contour, True), True)
            print("Smoothed")
            # Append the smoothed contour to the list if the contour has at least 4 vertices.
            if len(smooth_contour) <= MinVertices or len(smooth_contour) >= MaxVertices:
                continue

            area = cv.contourArea(smooth_contour)
            MINAREA = AREA / 1000
            MAXAREA = AREA / 2

            if not MINAREA < area < MAXAREA:
                continue

            smoothed_contours.append(smooth_contour)
            contourList.append(Contour(smooth_contour, originalImage, i))

            if self._logger:
                print(f"Smoothed contour {i}: ", smooth_contour)

        contour_image_smt = np.zeros_like(originalImage)
        cv.drawContours(contour_image_smt, smoothed_contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagainsmoothed.png', contour_image_smt)

        if self._logger:
            self._endTime = timer()
            print(f"Exiting ShapeFinder class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")
        return contourList

    def detectShapes2D(self, image, originalImage):
        if self._logger:
            self._startTime = timer()
            print("Entering ShapeFinder class, function detectShapes...")

        # Find contours and deal with them.
        contours = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # what is going on?
        contour_image = np.zeros_like(image)
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagain.jpg', contour_image)

        if self._logger:
            print("Contours have been found...")
            for i, contour in enumerate(contours):
                print(f"Countour {i}: ", contour)

            print("Trying to smoothen the countours...")
        # Smoothen contours - shapes at an angle might have noise.
        smoothed_contours = []
        # Higher epsilon means aggressive smoothing, low epsilon keeps more of the details.
        epsilon = 0.02

        contourList = []
        for i, contour in enumerate(contours):
            # Approximate the contour to smoothen it
            smooth_contour = cv.approxPolyDP(contour, epsilon * cv.arcLength(contour, True), True)

            # Append the smoothed contour to the list if the contour has at least 4 vertices.
            if len(smooth_contour) <= MinVertices or len(smooth_contour) >= MaxVertices:
                continue

            smoothed_contours.append(smooth_contour)
            contourList.append(Contour(smooth_contour, originalImage, i))

            if self._logger:
                print(f"Smoothed contour {i}: ", smooth_contour)

        contour_image_smt = np.zeros_like(image)
        cv.drawContours(contour_image_smt, smoothed_contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagainsmoothed.jpg', contour_image_smt)

        if self._logger:
            self._endTime = timer()
            print(f"Exiting ShapeFinder class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")
        return contourList