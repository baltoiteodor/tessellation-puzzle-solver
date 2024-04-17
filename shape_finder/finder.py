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

    def detectShapes(self, image, originalImage):
        if self._logger:
            self._startTime = timer()
            print("Entering ShapeFinder class, function detectShapes...")
            print(f"Image to be examined: {image}.")
        # Resize to better approximate shapes, this should work as the pieces have rough edges and shapes.
        # resizedImage = imutils.resize(image, width=300)
        # ratio = image.shape[0] / float(resizedImage.shape[0])

        # Adjust brightness and contrast
        # TODO: Decide if contrast needed.
        # alpha = 1.95
        # beta = 0
        # # No resizing for testing
        # contrastImage = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        cv.imwrite("imgContrastLab.png", image)

        # blurredImage = cv.GaussianBlur(image, (5, 5), 0)
        # cv.imwrite("blur.jpg", blurredImage)

        # grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # cv.imwrite("gray.png", grayImage)

        # equalizedImage = cv.equalizeHist(grayImage)
        # cv.imwrite("eq.png", equalizedImage)



        # Replaced blurredImage with grayImage.
        # threshImage = cv.threshold(grayImage, 105, 255, cv.THRESH_BINARY_INV)[1]
        #
        # cv.imwrite("thresh.jpg", threshImage)

        # Find contours and deal with them.
        edgeImage = cv.Canny(image, 10, 100)
        cv.imwrite("canny.png", edgeImage)

        contoursCV = cv.findContours(edgeImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contoursCV)

        # what is going on?
        contour_image = np.zeros_like(image)
        contourFiltered = np.zeros_like(image)
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagain.png', contour_image)
        filteredContours = []
        H, W = edgeImage.shape[:2]
        AREA = H * W
        for contour in contours:
            area = cv.contourArea(contour)

            # Area filtering:
            MINAREA = 2
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
        epsilon = 0.02

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

        contour_image_smt = np.zeros_like(image)
        cv.drawContours(contour_image_smt, smoothed_contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagainsmoothed.png', contour_image_smt)

        if self._logger:
            self._endTime = timer()
            print(f"Exiting ShapeFinder class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")
        return contourList
