# Parser for arguments and potentially flags. 

# Specify desired arguments.
import argparse
import json

import cv2
import cv2 as cv
import numpy as np

from preprocessor.PreProcessor import PreProcessor
from shape_finder.finder import ShapeFinder
from shape_processor.processor import Processor
from shape_rotation.rotator import Rotator
from puzzle_solver.solver import Solver

import matplotlib.pyplot as plt


def main():
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("-i", "--image", required=True,
                            help="path to input puzzle image.")  # Will correspond to "image" argument

    argsParser.add_argument("-log", "--logger", action="store_true", required=False,
                            help="Enable this for prints from all around the code")
    argsParser.add_argument("-logDetector", "--loggerDetector", action="store_true", required=False,
                            help="Enable this for prints from detector code")
    argsParser.add_argument("-logProcessor", "--loggerProcessor", action="store_true", required=False,
                            help="Enable this for prints from processor code")
    argsParser.add_argument("-logSolver", "--loggerSolver", action="store_true", required=False,
                            help="Enable this for prints from solver code")
    argsParser.add_argument("-logFinder", "--loggerFinder", action="store_true", required=False,
                            help="Enable this for prints from finder code")
    argsParser.add_argument("-logRotator", "--loggerRotator", action="store_true", required=False,
                            help="Enable this for prints from rotator code")
    argsParser.add_argument("-D", "--DLX", required=False,
                            help="Enable DLX algorithm for solving. 0 - self made; 1 - PyPy made; 2 - PyPy with optimisations.")
    argsParser.add_argument("-B", "--BKT", required=False,
                            help="Enable BKT algorithm for solving. 0 - no optimisation; "
                                 "1 - pieces rotate the optimal number of times and are pre-calculated when pieces are created.")
    argsParser.add_argument("-C", "--colour", action="store_true", required=False,
                            help="By enabling this option colours now matter in solving the puzzle.")
    argsParser.add_argument("-S", "--show", action="store_true", required=False,
                            help="By enabling this option you can visualise how the puzzle looks solved and how the pieces look.")
    argsParser.add_argument("-3D", "--3D", action="store_true", required=False,
                            help="By enabling this option you will use a different detection algorithm specialised for real world scenarios.")

    # Parse the arguments.
    args = vars(argsParser.parse_args())

    show = args["show"]
    # Set the logger flags to false
    allLog = args["logger"]
    detectorLog = args["loggerDetector"]
    processorLog = args["loggerProcessor"]
    solverLog = args["loggerSolver"]
    finderLog = args["loggerFinder"]
    rotatorLog = args["loggerRotator"]

    # Load image and send to shape recognition.         contours = cv.findContours(threshImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    originalImage = cv.imread(args["image"])

    copyImage = originalImage

    shapeFinder = ShapeFinder(finderLog | allLog)
    contours = []

    realProc = args["3D"]
    # Testing preproc:
    prep = PreProcessor(copyImage)
    if realProc:
        prep.applyContrast(1.5, -50)
        prep.applyBlur(5)
        prep.pyrMeanShiftFilterContours()
        prep.pyrMeanShiftFilterContours()
        # prep.applyContrast(1.25, 0)
        # prep.removeShadow()

        copyImage = prep.getImage()
        contours = shapeFinder.detectShapes3D(copyImage, originalImage)

    else:
        prep.basic2D()

        copyImage = prep.getImage()
        contours = shapeFinder.detectShapes2D(copyImage, originalImage)


    # Rotate the images in the case they are at an angle.
    rotator = Rotator(rotatorLog | allLog)
    rotatedContours = rotator.rotate(contours, originalImage)

    # Find the units in which to break the shapes.
    # Grid the smallest rectangle in a grid with units lxl, where l is a divisor of the smallest side.
    # Look for the biggest l s.t. the area lost in the process is less than a given percent.
    processor = Processor(rotatedContours, processorLog | allLog)
    lMax = processor.findUnit(originalImage)
    pieces = processor.getPieces()

    if show:
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

    colour = args["colour"]

    bkt = -1
    if args["BKT"] is not None:
        bkt = int(args["BKT"])

    dlx = -1
    if args["DLX"] is not None:
        dlx = int(args["DLX"])
    puzzleSolver = Solver(solverLog | allLog, originalImage, bkt, dlx, colour)

    if puzzleSolver.solveBackTracking(pieces):
        if show:
            rgbArray = np.array(puzzleSolver.getSolution()).astype(np.uint8)

            # Display the RGB array using Matplotlib
            plt.imshow(rgbArray)
            plt.axis('off')  # Turn off axis labels
            plt.show()
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
    # piecesList = [[int(cell) for cell in row] for arr in pieces for row in arr]
    # array_of_2d_arrays_as_lists = [arr.tolist() for arr in pieces]
    #
    # with open('pieces.json', 'w') as piecesFile:
    #     json.dump(array_of_2d_arrays_as_lists, piecesFile, indent=2)


if __name__ == '__main__':
    main()