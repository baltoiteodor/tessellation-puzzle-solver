# Parser for arguments and potentially flags. 

# Specify desired arguments.
import argparse
import json

import cv2 as cv

from shape_finder.finder import ShapeFinder
from shape_processor.processor import Processor
from shape_rotation.rotator import Rotator
from puzzle_solver.solver import Solver

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

#
##
#

# First, we will write an algorithm that solves this puzzle in Python in order to afterward compare it to a C++ puzzle_solver
# and get info on the rate of improvement.

#
##
#

# Python puzzle puzzle_solver.

puzzleSolver = Solver()
if puzzleSolver.solveBackTracking(pieces):
    print("Puzzle is solvable.")
else:
    print("Something is or went wrong with the puzzle.")


#
##
#

# The C++ approach begins here.

#
##
#

# To transfer the pieces to the C++ Solver, we write our pieces to a file from where the puzzle_solver will read them and
# continue. We will use JSON format.

# Convert np arrays to python lists for serialisation.
piecesList = [[int(cell) for cell in row] for arr in pieces for row in arr]
array_of_2d_arrays_as_lists = [arr.tolist() for arr in pieces]

with open('pieces.json', 'w') as piecesFile:
    json.dump(array_of_2d_arrays_as_lists, piecesFile, indent=2)
