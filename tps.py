# Parser for arguments and potentially flags. 

# Specify desired arguments.
import argparse

import cv2 as cv

from shape_finder.finder import ShapeFinder
from shape_processor.processor import Processor
from shape_rotation.rotator import Rotator


argsParser = argparse.ArgumentParser()
argsParser.add_argument("-i", "--image", required = True, 
                        help = "path to input puzzle image.") # Will correspond to "image" argument

# Parse the arguments.
args = vars(argsParser.parse_args())


# Load image and send to shape recognition.         contours = cv.findContours(threshImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# This will then return the list of corners of the detected polygons.
image = cv.imread(args["image"])
shapeFinder = ShapeFinder()
contours = shapeFinder.detectShapes(image)
print(contours)


# Rotate the images in the case they are at an angle.
rotator = Rotator()
rotatedContours = rotator.rotate(contours, image)


# Find the units in which to break the shapes. 
# Grid the smallest rectangle in a grid with units lxl, where l is a divisor of the smallest side. 
# Look for the biggest l s.t. the area lost in the process is less than a given percent. 
processor = Processor(rotatedContours)
lMax = processor.findUnit()
pieces = processor.getPieces() 

print("Here are the grids for the pieces:")
print(pieces)


