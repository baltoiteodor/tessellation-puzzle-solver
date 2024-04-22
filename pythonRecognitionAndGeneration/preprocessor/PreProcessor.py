import cv2
import imutils
import numpy as np
# import skimage.filters as filters


class PreProcessor:
    def __init__(self, image):
        self._image = image

    def getImage(self):
        return self._image

    def getSaturation(self):
        # LAB image coming here.
        # hlsImage = cv2.cvtColor(self._image, cv2.COLOR_BGR2HLS)
        saturationChannel = self._image[:, :, 1]
        cv2.imwrite("saturation.png", saturationChannel)

    def applyContrast(self, alpha, beta):
        self._image = cv2.convertScaleAbs(self._image, alpha=alpha, beta=beta)
        cv2.imwrite("imgContrast.png", self._image)

    def applyBlur(self, blur):
        self._image = cv2.GaussianBlur(self._image, (blur, blur), 0)
        cv2.imwrite("blur.png", self._image)

    def gray(self):
        self._image = cv2.cvtColor(self._image, cv2.COLOR_LAB2BGR)
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("grayed.png", self._image)

    def differentGray(self):
        image_float = self._image.astype(np.float32) / 255.0

        # Compute the grayscale intensity using the luminosity method
        gray = np.dot(image_float[..., :3], [0.299, 0.587, 0.114])

        # Scale the intensity values to the range [0, 255] and convert to uint8
        gray_uint8 = (gray * 255).astype(np.uint8)
        self._image = gray_uint8
        cv2.imwrite("diffGray.png", gray_uint8)

    def differentGray2(self):
        image_lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)

        L_channel = image_lab[:, :, 0]
        a_channel = image_lab[:, :, 1]
        b_channel = image_lab[:, :, 2]

        # Convert LAB channels to grayscale using the formula
        gray_image = 0.2126 * L_channel + 0.7152 * a_channel + 0.0722 * b_channel

        # Scale the grayscale image to the range [0, 255] and convert to uint8
        gray_image_uint8 = ((gray_image / np.max(gray_image)) * 255).astype(np.uint8)
        cv2.imwrite("diffGray2.png", gray_image_uint8)
        self._image = gray_image_uint8
        return gray_image_uint8

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

    def pyrMeanShiftFilter(self):
        # filteredImage = cv2.bilateralFilter(self._image, 30, 80, 80)
        labImage = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        filteredImage = cv2.pyrMeanShiftFiltering(labImage, 50, 20)
        cv2.imwrite("pyr.png", filteredImage)
        self._image = filteredImage

    def lab(self):
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        cv2.imwrite("lab.png", self._image)

    def bilateralFilter(self):
        self._image = cv2.bilateralFilter(self._image, 10, 45, 45)
        cv2.imwrite("bilateral.png", self._image)

    # def guidedFilter(self):
    #     guided_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
    #
    #     # Apply guided filter
    #     filtered_image = cv2.ximgproc.guidedFilter(guided_image, guided_image, radius=30, eps=0.1)
    #     cv2.imwrite("guide.png", filtered_image)
    #     self._image = filtered_image
    #
    # def division(self, blur):
    #     smooth = cv2.GaussianBlur(self._image, (blur, blur), 0)
    #
    #     # divide gray by morphology image
    #     division = cv2.divide(self._image, smooth, scale=255)
    #     cv2.imwrite('division.png', division)
    #
    #     # sharpen using unsharp masking
    #     sharp = filters.unsharp_mask(division, radius=1.5, amount=1.5, preserve_range=False)
    #     sharp = (255 * sharp).clip(0, 255).astype(np.uint8)
    #     self._image = sharp

    def hueChannel(self):
        hsv_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]
        cv2.imwrite("hueChannel.png", hue_channel)
        self._image = hue_channel

    def morphologicalOpen(self):
        kernel = np.ones((15, 15), np.uint8)
        opening = cv2.morphologyEx(self._image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("openingmorph.png", opening)
        cv2.imwrite("closemor.png", closing)
        self._image = closing

    def adaptiveThreshold(self, blockSize, C):
        self._image = cv2.adaptiveThreshold(self._image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            blockSize, C)
        cv2.imwrite("adaptiveThresh.png", self._image)

    def otsu(self):
        _, self._image = cv2.threshold(self._image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("otsu.png", self._image)

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
