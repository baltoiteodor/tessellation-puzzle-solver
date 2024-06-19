import cv2
import numpy as np

from misc.types import *


# Receives an openCV contour, construct an object that adds a colour aspect to it.
class Contour:
    def __init__(self, contour, image, ordNum: int):
        self._contour = self._originalContour = contour
        self._image = image
        self._ordNum = ordNum
        self._initialize()
        self._rotated = False
        self._lowestPoint = None
        self._angle = 0
        self._colour = self._calculateColour()


    def rotateImage(self, image, angle):
        # Determine the center.
        (height, width) = image.shape[:2]
        (centerX, centerY) = (width // 2, height // 2)

        # Grab the rotation matrix, then grab the sine and cosine and compute new dimensions.
        matrix = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])

        newWidth = int((height * sin) + (width * cos))
        newHeight = int((height * cos) + (width * sin))

        # Adjust the rotation matrix to take into account translation.
        matrix[0, 2] += (newWidth / 2) - centerX
        matrix[1, 2] += (newHeight / 2) - centerY

        # Perform rotation and return the image.
        return cv2.warpAffine(image, matrix, (newWidth, newHeight))

    # This function is used in the suboptimal method of recreating solutions. Can be used for comparisons.
    def createROIAtAngle(self, targetLocation, targetImg, rotationDegrees):
        originalImg = self._image

        # Create a mask from the contour and roi.
        mask = np.zeros(originalImg.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self._originalContour], -1, 255, -1)

        roi = cv2.bitwise_and(originalImg, originalImg, mask=mask)

        # Calculate the bounding box of the contour to extract masks.

        x, y, w, h = cv2.boundingRect(self._originalContour)
        extractedShape = roi[y:y+h, x:x+w]
        shapeMask = mask[y:y+h, x:x+w]

        # Rotate the extracted shape and the mask.
        rotatedShape = self.rotateImage(extractedShape, rotationDegrees)
        rotatedMask = self.rotateImage(shapeMask, rotationDegrees).astype(np.uint8)

        height, width = rotatedShape.shape[:2]
        targetX, targetY = targetLocation

        # Ensure the piece is in bounds.
        if targetX + width > targetImg.shape[1] or targetY + height > targetImg.shape[0]:
            raise ValueError("Piece goes out of bounds of the target image.")

        # Mask target region and add the piece.
        targetRegion = targetImg[targetY:targetY+height, targetX:targetX+width]
        targetRegionMasked = cv2.bitwise_and(targetRegion, targetRegion, mask=cv2.bitwise_not(rotatedMask))
        finalRegion = cv2.add(targetRegionMasked, rotatedShape)
        targetImg[targetY:targetY+height, targetX:targetX+width] = finalRegion

        return targetImg
    def getOrdNum(self):
        return self._ordNum

    def getColour(self):
        return self._colour

    def getContour(self):
        return self._contour

    def getBoundingRect(self):
        return self._boundingRectangle

    def setContour(self, contour):
        self._contour = contour
        self._initialize()

    def getArea(self):
        return self._area

    def getMinAreaRect(self):
        return self._minAreaRect

    def _initialize(self):
        self._boundingRectangle = cv2.boundingRect(self._contour)
        self._minAreaRect = cv2.minAreaRect(self._contour)
        self._area = cv2.contourArea(self._contour)

    # Sample 11 points, return the median value surrounding neighbours.
    def _calculateColour(self):
        contour = self._contour
        x, y, width, height = self._boundingRectangle
        noPoints = 11

        randomPoint = (np.random.randint(x, x + width),
                       np.random.randint(y, y + height))
        blue = []
        green = []
        red = []
        for _ in range(noPoints):
            isInside = cv2.pointPolygonTest(contour, randomPoint, measureDist=False)
            while isInside <= 0:
                randomPoint = (np.random.randint(x, x + width),
                               np.random.randint(y, y + height))
                isInside = cv2.pointPolygonTest(contour, randomPoint, measureDist=False)
            # Found a point that is inside.
            b, g, r = self._image[randomPoint[1], randomPoint[0]]

            blue.append(b)
            green.append(g)
            red.append(r)

        # Find the median value from the intensities of the random pixels and return it.
        medianB = np.median(blue)
        medianG = np.median(green)
        medianR = np.median(red)

        return medianR, medianG, medianB

    # If the contour was not rotated then the point remains the same.
    # Otherwise, we rotate the point.
    def getOriginalCoord(self, point):
        if not self._rotated:
            return point
        # Apply the same function on the point, but with opposite angle.
        x, y = point
        transX = x - self._lowestPoint[0][0]
        transY = y - self._lowestPoint[0][1]

        theta, rho = cart2pol(transX, transY)

        theta = np.rad2deg(theta)
        theta = (theta - self._angle) % 360
        theta = np.deg2rad(theta)

        newX, newY = pol2cart(theta, rho)

        originalX = (newX + x).astype(np.int32)
        originalY = (newY + y).astype(np.int32)

        return originalX, originalY

    def rotate(self, point, angle: float):
        self._rotated = True
        self._angle = angle
        self._lowestPoint = point
        pointY = point[0][1]
        pointX = point[0][0]

        # translate to origin.
        cntTrans = self._contour - [pointX, pointY]

        coordinates = cntTrans[:, 0, :]
        xs, ys = coordinates[:, 0], coordinates[:, 1]
        thetas, rhos = cart2pol(xs, ys)

        thetas = np.rad2deg(thetas)
        thetas = (thetas + angle) % 360
        thetas = np.deg2rad(thetas)

        xs, ys = pol2cart(thetas, rhos)

        cntTrans[:, 0, 0] = xs
        cntTrans[:, 0, 1] = ys

        cntRotated = cntTrans + [pointX, pointY]
        cntRotated = cntRotated.astype(np.int32)
        return cntRotated

    def getOriginalContour(self):
        return self._originalContour

    def getImage(self):
        return self._image

    def __repr__(self):
        return f"Contour {self._ordNum} with colour " \
               f"{self._colour} looks like this: \n {self._contour}"


Contours = List[Contour]


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
