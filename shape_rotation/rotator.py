import cv2 as cv
import numpy as np

thresholdAngle = 3


class Rotator():
    def __init__(self):
        pass

    def rotate(self, contours, image):
        rotatedImage = np.zeros_like(image)
        rotatedContours = []
        for c in contours:
            # Make the contour straight using the min area rectangle.
            rect = cv.minAreaRect(c)

            angle = rect[2]
            print("Angle: ", angle)
            # rotate if at an angle
            if abs(angle) > thresholdAngle and abs(abs(angle) - 90) > thresholdAngle:
                rows, cols = rotatedImage.shape[:2]
                M = cv.getRotationMatrix2D(rect[0], angle, 1)

                contour_rotated = cv.transform(c.reshape(-1, 1, 2), M).reshape(-1, 2)
                contour_rotated = contour_rotated.astype(np.int32)
                rotatedContours.append(contour_rotated)
                # Needed for debugging
                cv.drawContours(rotatedImage, [contour_rotated], 0, (255, 255, 255), 2)

                print(c)
                print("Here is the same shape, rotated: ")
                print(contour_rotated)

                ogArea = cv.contourArea(c)
                rotatedArea = cv.contourArea(contour_rotated)
                print("Area:")
                print(ogArea)
                print(rotatedArea)
            else:
                rotatedContours.append(c)
                # Needed for debugging
                cv.drawContours(rotatedImage, [c], 0, (255, 255, 255), 2)

        cv.imwrite("straight.jpg", rotatedImage)
        return rotatedContours
