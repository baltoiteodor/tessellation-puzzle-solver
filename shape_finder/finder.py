import cv2 as cv
import imutils
import cv2 as cv
import numpy as np


class ShapeFinder():
    def __init__ (self): 
        pass 
    def detectShapes(self, image): 
        # Resize to better approximate shapes, this should work as the pieces have rough edges and shapes.
        # resizedImage = imutils.resize(image, width=300)
        # ratio = image.shape[0] / float(resizedImage.shape[0])

        # Adjust brightness and contrast
        # TODO: make this automatic
        alpha = 1.95
        beta = 0
        # No resizing for testing
        contrastImage = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        cv.imwrite("contrast.jpg", contrastImage)

        # Convert resized image to GS, Blur it and apply threshold.
        grayImage = cv.cvtColor(contrastImage, cv.COLOR_BGR2GRAY)
        cv.imwrite("gray.jpg", grayImage)


        # blurredImage = cv.GaussianBlur(grayImage, (5, 5), 0)
        # cv.imwrite("blur.jpg", blurredImage)

        # Replaced blurredImage with grayImage.
        threshImage = cv.threshold(grayImage, 240, 255, cv.THRESH_BINARY_INV)[1]

        cv.imwrite("thresh.jpg", threshImage)

        # Find contours and deal with them.
        contours = cv.findContours(threshImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # what is going on?
        contour_image = np.zeros_like(image)
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv.imwrite('contoursagain.jpg', contour_image)
        for contour in contours:
            print("Original Contour: ", contour)

        # Smoothen contours - shapes at an angle might have noise.
        smoothed_contours = []
        # Higher epsilon means aggressive smoothing, low epsilon keeps more of the details.
        epsilon = 0.01
        for contour in contours:
            # Approximate the contour to smoothen it
            smooth_contour = cv.approxPolyDP(contour, epsilon * cv.arcLength(contour, True), True)

            # Append the smoothed contour to the list
            smoothed_contours.append(smooth_contour)
        return smoothed_contours