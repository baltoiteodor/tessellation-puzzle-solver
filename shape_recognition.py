from image_search.detector import Detector

import argparse
import imutils
import cv2 as cv
import numpy as np

# Parser for arguments and potentially flags. 

# Specify desired arguments.
argsParser = argparse.ArgumentParser()
argsParser.add_argument("-i", "--image", required = True, 
                        help = "path to input puzzle image.") # Will correspond to "image" argument

# Parse the arguments.
args = vars(argsParser.parse_args())


# Loading and Image Processing. 

image = cv.imread(args["image"])

# Resize to better approximate shapes, this should work as the pieces have rough edges and shapes.
resizedImage = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resizedImage.shape[0])

# Adjust brightness and contrast
# TODO: make this automatic
alpha = 1.95
beta = 0 
contrastImage = cv.convertScaleAbs(resizedImage, alpha=alpha, beta=beta)
cv.imwrite("contrast.jpg", contrastImage)

# Convert resized image to GS, Blur it and apply threshold.
grayImage = cv.cvtColor(contrastImage, cv.COLOR_BGR2GRAY)
cv.imwrite("gray.jpg", grayImage)


# blurredImage = cv.GaussianBlur(grayImage, (5, 5), 0)
# cv.imwrite("blur.jpg", blurredImage)

# Replaced blurredImage with grayImage.
threshImage = cv.threshold(grayImage, 195, 255, cv.THRESH_BINARY_INV)[1]

cv.imwrite("thresh.jpg", threshImage)

# Find contours and deal with them.
contours = cv.findContours(threshImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
   
detector = Detector()

rotatedImage = np.zeros_like(image)
for c in contours: 
    print("Here is a shape: ")
    detector.detect(c)
    # Make the contour straight using the min area rectangle.
    rect = cv.minAreaRect(c)
    
    angle = rect[2]
    rows, cols = rotatedImage.shape[:2]
    M = cv.getRotationMatrix2D(rect[0], angle, 1)

    contour_rotated = cv.transform(c.reshape(-1, 1, 2), M).reshape(-1, 2)
    contour_rotated = contour_rotated.astype(np.int32)

    # Needed for debugging
    cv.drawContours(rotatedImage, [contour_rotated], 0, (255,255,255), 2)

    print("Here is the same shape, rotated: ")
    detector.detect(contour_rotated)

    ogArea = cv.contourArea(c)
    rotatedArea = cv.contourArea(contour_rotated)
    print("Area:")
    print(ogArea)
    print(rotatedArea)

cv.imwrite("straight.jpg", rotatedImage)


