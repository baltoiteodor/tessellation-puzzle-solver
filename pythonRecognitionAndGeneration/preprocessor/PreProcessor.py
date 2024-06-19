import cv2
import imutils
import numpy as np
from timeit import default_timer as timer


class PreProcessor:
    def __init__(self, image):
        self._image = image
        self._startTime = self._endTime = 0

    def getImage(self):
        return self._image

    def getSaturation(self):
        # LAB image coming here.
        saturationChannel = self._image[:, :, 1]
        # Uncomment to see output.
        # cv2.imwrite("saturation.png", saturationChannel)

    def applyContrast(self, alpha, beta):
        self._image = cv2.convertScaleAbs(self._image, alpha=alpha, beta=beta)
        # Uncomment to see output.
        # cv2.imwrite("imgContrast.png", self._image)

    def applyBlur(self, blur):
        self._image = cv2.GaussianBlur(self._image, (blur, blur), 0)
        # Uncomment to see output.
        # cv2.imwrite("blur.png", self._image)

    def gray(self):
        # If LAB image comes in.
        self._image = cv2.cvtColor(self._image, cv2.COLOR_LAB2BGR)
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        # Uncomment to see output.
        # cv2.imwrite("grayed.png", self._image)

    def differentGray(self):
        # Different way of calculating the grayscale.
        imageFloat = self._image.astype(np.float32) / 255.0

        # Compute the grayscale intensity using the luminosity method.
        gray = np.dot(imageFloat[..., :3], [0.299, 0.587, 0.114])

        # Scale the intensity values to the range [0, 255].
        grayInt = (gray * 255).astype(np.uint8)
        self._image = grayInt

        # Uncomment to see output.
        # cv2.imwrite("diffGray.png", grayInt)

    def differentGray2(self):
        # Different way of calculating the grayscale.
        imageLAB = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)

        LChannel = imageLAB[:, :, 0]
        aChannel = imageLAB[:, :, 1]
        bChannel = imageLAB[:, :, 2]

        # Convert LAB channels to grayscale using the formula.
        grayImage = 0.2126 * LChannel + 0.7152 * aChannel + 0.0722 * bChannel

        # Scale the grayscale image to the range [0, 255] and convert to int.
        grayImageInt = ((grayImage / np.max(grayImage)) * 255).astype(np.uint8)
        # Uncomment to see output.
        # cv2.imwrite("diffGray2.png", grayImageInt)

        self._image = grayImageInt
        return grayImageInt


    def pyrMeanShiftFilter(self):
        labImage = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        # filteredImage = cv2.pyrMeanShiftFiltering(labImage, 50, 20) This was good for some reason
        filteredImage = cv2.pyrMeanShiftFiltering(labImage, 15, 25)
        # Uncomment to inspect result.
        # cv2.imwrite("pyr.png", filteredImage)
        self._image = filteredImage

    def lab(self):
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        # Uncomment to see result.
        # cv2.imwrite("lab.png", self._image)

    def bilateralFilterApply(self):
        self._image = cv2.bilateralFilter(self._image, 10, 45, 45)
        # Uncomment to see output.
        # cv2.imwrite("bilateral.png", self._image)

    def hueChannel(self):
        hsvImage = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        hueChannel = hsvImage[:, :, 0]
        # Uncomment to see output.
        # cv2.imwrite("hueChannel.png", hueChannel)
        self._image = hueChannel

    def morphologicalOpen(self):
        kernel = np.ones((15, 15), np.uint8)
        opening = cv2.morphologyEx(self._image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # Uncomment to see results.
        # cv2.imwrite("openingmorph.png", opening)
        # cv2.imwrite("closemor.png", closing)
        self._image = closing

    def adaptiveThreshold(self, blockSize, C):
        self._image = cv2.adaptiveThreshold(self._image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            blockSize, C)
        # Uncomment to see outcome.
        # cv2.imwrite("adaptiveThresh.png", self._image)

    def otsu(self):
        _, self._image = cv2.threshold(self._image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Uncomment to see output.
        # cv2.imwrite("otsu.png", self._image)

    def basic2D(self):
        self._startTime = timer()

        alpha = 1.95
        beta = 0
        contrastImage = cv2.convertScaleAbs(self._image, alpha=alpha, beta=beta)

        # Uncomment for result.
        # cv2.imwrite("contrast.jpg", contrastImage)

        # Convert resized image to GS and apply threshold.
        grayImage = cv2.cvtColor(contrastImage, cv2.COLOR_BGR2GRAY)

        threshImage = cv2.threshold(grayImage, 240, 255, cv2.THRESH_BINARY_INV)[1]

        # Uncomment to see results.
        # cv2.imwrite("thresh.jpg", threshImage)

        self._image = threshImage

        self._endTime = timer()

    def jigsaw2DV2(self):
        self._startTime = timer()
        topLeftColour = self._image[0, 0]

        # Create a mask for the background based on colour.
        mask = cv2.inRange(self._image, topLeftColour, topLeftColour)

        # Invert the mask to get the foreground.
        invertedMask = cv2.bitwise_not(mask)

        # Create a black image for the background and white for foreground and combine them.
        background = np.zeros_like(self._image)
        foreground = np.full_like(self._image, 255)

        result = cv2.bitwise_and(foreground, foreground, mask=invertedMask)
        result += cv2.bitwise_and(background, background, mask=mask)

        grayResult = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        self._image = grayResult

        # Uncomment to see the pieces detected.
        # cv2.imwrite("cema.png", self._image)

        self._endTime = timer()

    def getTimeTaken(self):
        return self._endTime - self._startTime
