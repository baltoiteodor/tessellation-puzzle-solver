import cv2
import imutils
import numpy as np


class PreProcessor:
    def __init__(self, image):
        self._image = image

    def getImage(self):
        return self._image

    def applyContrast(self):
        alpha = 1.25
        beta = -25.0
        self._image = cv2.convertScaleAbs(self._image, alpha=alpha, beta=beta)
        cv2.imwrite("imgContrast.png", self._image)

    def removeShadow(self):
        # Convert the image to the LAB color space
        labImage = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into its channels
        L, A, B = cv2.split(labImage)

        # Threshold the L channel to identify potential shadow regions
        _, shadowMask = cv2.threshold(L, 55, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to refine the shadow mask
        kernel = np.ones((5, 5), np.uint8)
        shadowMask = cv2.morphologyEx(shadowMask, cv2.MORPH_CLOSE, kernel)

        # Invert the shadow mask to obtain a mask for the non-shadow regions
        nonShadowMask = cv2.bitwise_not(shadowMask)

        # Apply the non-shadow mask to the original image
        resultImage = cv2.bitwise_and(self._image, self._image, mask=nonShadowMask)

        cv2.imwrite("noshadows.png", resultImage)
        self._image = resultImage
        # return resultImage

    def pyrMeanShiftFilterContours(self):
        imgCopy = self._image.copy()
        # filteredImage = cv2.bilateralFilter(self._image, 30, 80, 80)
        labImage = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        filteredImage = cv2.pyrMeanShiftFiltering(labImage, 20, 30)
        grayImage = cv2.cvtColor(filteredImage, cv2.COLOR_BGR2GRAY)
        edgeImage = cv2.Canny(grayImage, 50, 100)
        cv2.imwrite("bilateral.png", edgeImage)

        contours = cv2.findContours(edgeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        H, W = self._image.shape[:2]
        AREA = H * W
        for cnt in contours:
            # print(cnt)
            area = cv2.contourArea(cnt)
            if not AREA / 1000 < area < AREA / 5:
                continue
            if len(cnt) < 20:
                continue
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            cv2.drawContours(imgCopy, [box], -1, (255, 0, 255), 2, cv2.LINE_AA)

        result = np.hstack((self._image, filteredImage, cv2.cvtColor(edgeImage, cv2.COLOR_GRAY2BGR), imgCopy))
        cv2.imwrite("result.png", result)
        self._image = filteredImage

    def basic2D(self):
        alpha = 1.95
        beta = 0
        # No resizing for testing
        contrastImage = cv2.convertScaleAbs(self._image, alpha=alpha, beta=beta)
        cv2.imwrite("contrast.jpg", contrastImage)

        # Convert resized image to GS, Blur it and apply threshold.
        grayImage = cv2.cvtColor(contrastImage, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("gray.jpg", grayImage)

        # blurredImage = cv.GaussianBlur(grayImage, (5, 5), 0)
        # cv.imwrite("blur.jpg", blurredImage)

        # Replaced blurredImage with grayImage.
        threshImage = cv2.threshold(grayImage, 240, 255, cv2.THRESH_BINARY_INV)[1]

        cv2.imwrite("thresh.jpg", threshImage)

        self._image = threshImage
