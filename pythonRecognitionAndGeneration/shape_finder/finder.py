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
        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)
            # Area filtering:
            MINAREA = AREA / 600
            MAXAREA = AREA / 2

            # print(area)
            # print(contour)

            # image = np.zeros((800, 800), dtype=np.uint8)
            # image2 = np.zeros((800, 800), dtype=np.uint8)
            # min_x = np.min(contour[:, :, 0])
            # min_y = np.min(contour[:, :, 1])

            # translated_contour = contour - [min_x - 20, min_y - 20]
            # area = cv.contourArea(contour)
            # print(area)
            # print(cv.pointPolygonTest(translated_contour, (200, 200), False))
            # cv.drawContours(image, [translated_contour], -1, (255, 255, 255), 2)
            # cv.circle(image, (200, 200), radius=5, color=(255, 255, 255), thickness=-1)
            # cv.imshow("H", image)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # simplified_contour = cv.approxPolyDP(contour, 0.5, True)
            # translated_contourV2 = simplified_contour - [min_x - 20, min_y - 20]

            # print(cv.contourArea(simplified_contour))
            # cv.drawContours(image2, [translated_contourV2], -1, (255, 255, 255), 2)
            # cv.imshow("H", image2)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            if not MINAREA < area < MAXAREA:
                continue

            contour = Contour(cv.approxPolyDP(contour, 0.3, True), originalImage, i)
            filteredContours.append(contour)

        # cv.drawContours(contourFiltered, filteredContours, -1, (0, 255, 0), 2)
        # cv.imwrite('contoursagainButFiltered.png', contourFiltered)

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
            # print("smoothing")
            smooth_contour = cv.approxPolyDP(contour, epsilon * cv.arcLength(contour, True), True)
            # print("Smoothed")
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
        # image = 255 - image
        cv.imwrite("imgContrastLab.png", image)
        # edgeImage = cv.Canny(image, 10, 150)
        # cv.imwrite("canny.png", edgeImage)

        contoursCV = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contoursCV)

        # what is going on?
        contour_image = np.zeros_like(originalImage)
        contourFiltered = np.zeros_like(originalImage)
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagain.png', contour_image)
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

        cv.drawContours(contourFiltered, clt, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagainButFiltered.png', contourFiltered)

        if self._logger:
            print("Contours have been found and triaged...")
            for i, contour in enumerate(contours):
                print(f"Countour {i}: ", contour)

            # print("Trying to smoothen the countours...")
        # Smoothen contours - shapes at an angle might have noise.
        # smoothed_contours = []
        # Higher epsilon means aggressive smoothing, low epsilon keeps more of the details.
        # epsilon = 0.01

        # contourList = []
        # for i, contour in enumerate(contours):
        #     # Approximate the contour to smoothen it
        #     print("smoothing")
        #     smooth_contour = cv.approxPolyDP(contour, epsilon * cv.arcLength(contour, True), True)
        #     print("Smoothed")
        #     # Append the smoothed contour to the list if the contour has at least 4 vertices.
        #     if len(smooth_contour) <= MinVertices or len(smooth_contour) >= MaxVertices:
        #         continue
        #
        #     area = cv.contourArea(smooth_contour)
        #     MINAREA = AREA / 1000
        #     MAXAREA = AREA / 2
        #
        #     if not MINAREA < area < MAXAREA:
        #         continue
        #
        #     smoothed_contours.append(smooth_contour)
        #     contourList.append(Contour(smooth_contour, originalImage, i))
        #
        #     if self._logger:
        #         print(f"Smoothed contour {i}: ", smooth_contour)
        #
        # contour_image_smt = np.zeros_like(originalImage)
        # cv.drawContours(contour_image_smt, smoothed_contours, -1, (0, 255, 0), 2)
        # cv.imwrite('contoursagainsmoothed.png', contour_image_smt)
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
        # print()
        # what is going on?
        contour_image = np.zeros_like(originalImage)
        cv.drawContours(contour_image, contours, -1, (255, 255, 255), thickness = 1)
        cv.imwrite('contoursagain.jpg', contour_image)
        # cv.imshow('Contours', contour_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

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

        contour_image_smt = np.zeros_like(originalImage)
        cv.drawContours(contour_image_smt, smoothed_contours, -1, (255, 255, 255), thickness = 1)
        cv.imwrite('contoursagainsmoothed.jpg', contour_image_smt)
        # cv.imshow('Contours', contour_image_smt)
        # cv.waitKey(0)
        self._endTime = timer()
        if self._logger:
            print(f"Exiting ShapeFinder class: {self._endTime - self._startTime}...")
            print("---")
            print("----------------------------")
            print("---")
        return contourList

    def getTimeTaken(self):
        return self._endTime - self._startTime