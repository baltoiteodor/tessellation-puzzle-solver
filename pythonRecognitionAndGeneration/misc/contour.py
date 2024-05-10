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

    # def setUpColourMap(self, rotatedContour):
    #     r, g, b = self._colour
    #     rotatedImage = np.zeros_like(self._image)
    #     cv2.drawContours(rotatedImage, rotatedContour, -1, (b, g, r), thickness=cv2.FILLED)
    #
    #     # mask the region inside the rotated contour.
    #     mask = np.zeros_like(self._image)
    #     cv2.drawContours(mask, [self._contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    #
    #     # Transfer colors from original image to rotated image within the masked region
    #     resultImage = np.where(mask != 0, self._image, rotatedImage)
    #
    #     return resultImage
    def createROI(self, targetLocation, targetImage):
        targetSize = targetImage.shape[:2]

        # Create a mask from the contour
        mask = np.zeros(self._image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self._originalContour], -1, 255, -1)  # Draw filled contour

        # Create the ROI from the mask on the original image
        roi = cv2.bitwise_and(self._image, self._image, mask=mask)

        # Calculate the bounding box of the contour to extract the minimal area
        x, y, w, h = cv2.boundingRect(self._originalContour)

        # Extract the region of interest using the bounding rectangle
        extractedShape = roi[y:y+h, x:x+w]

        # Prepare the mask for this shape to be used in the target image
        shapeMask = mask[y:y+h, x:x+w]

        # Location to place the shape on the target image
        tx, ty = targetLocation

        # Ensure the target location does not go out of bounds
        if tx + w > targetSize[1] or ty + h > targetSize[0]:
            raise ValueError("Target location goes out of bounds of the target image.")

        # Prepare the region on the target image where the shape will be placed
        targetRegion = targetImage[ty:ty+h, tx:tx+w]
        # Mask out the area in the target image
        print("mortimati: ", targetLocation)
        print("uwu 2 ", extractedShape.shape)
        print("uwu 3 ", shapeMask.shape)
        print("uwu ffs ", targetRegion.shape, w)

        targetRegionMasked = cv2.bitwise_and(targetRegion, targetRegion, mask=cv2.bitwise_not(shapeMask))
        print("uwu ", targetRegionMasked.shape)

        # Combine the extracted shape with the target region
        finalRegion = cv2.add(targetRegionMasked, extractedShape)

        # Place the combined region back into the target image
        targetImage[ty:ty+h, tx:tx+w] = finalRegion

        cv2.imwrite(f'mata{self._ordNum}.png', targetImage)
        return targetImage
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

    # Sample 11 points, return the median value of an average of surrounding
    # neighbours.
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
            # Found a point that is inside, TODO: make up an average of neighbours.
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
