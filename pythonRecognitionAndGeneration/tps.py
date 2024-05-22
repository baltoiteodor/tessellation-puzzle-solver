# Parser for arguments and potentially flags. 

# Specify desired arguments.
import argparse
import json

import cv2
import cv2 as cv
import numpy as np
from timeit import default_timer as timer

from preprocessor.PreProcessor import PreProcessor
from shape_finder.finder import ShapeFinder
from shape_processor.processor import Processor
from shape_rotation.rotator import Rotator
from puzzle_solver.solver import Solver
from puzzle_solver.helper import *

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
    argsParser.add_argument("-cpp", "--cpp", action="store_true", required=False,
                            help="By enabling this option the solver will use the C++ implementation of the DLX algorithm.")

    argsParser.add_argument("-jigsaw", "--jigsaw", action="store_true", required=False,
                            help="Specify that the input puzzle is a basic jigsaw puzzle. Expect a possibly faulty solution to the problem :).")

    # Disable scaling in tessellation mode.
    argsParser.add_argument("-noScaling", "--noScaling", action="store_true", required=False,
                            help="Specify that the input tessellation board is 1:1 with the pieces.")

    # Rows and columns number of pieces for jigsaw.
    argsParser.add_argument("-r", "--rows", required=False,
                            help="Jigsaw pieces per column.")
    argsParser.add_argument("-c", "--columns", required=False,
                            help="Jigsaw pieces per row.")

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
    jigsaw = args["jigsaw"]
    if args["rows"] is not None:
        rows = int(args["rows"])
    if args["columns"] is not None:
        columns = int(args["columns"])
    # print(rows, columns)
    # Testing preproc:
    prep = PreProcessor(copyImage)
    if realProc:

        #   Testing thresh:
        # prep.applyContrast(1.5, 0)
        # prep.removeShadow()
        prep.morphologicalOpen()

        prep.applyBlur(53)
        prep.applyContrast(2.0, -50)

        # prep.hueChannel()
        # prep.lab()
        # prep.bilateralFilter()
        prep.morphologicalOpen()
        # prep.morphologicalOpen()

        prep.pyrMeanShiftFilter()
        # prep.pyrMeanShiftFilter()
        # prep.bilateralFilter()
        # prep.guidedFilter()
        # prep.pyrMeanShiftFilterContours()
        # prep.pyrMeanShiftFilterContours()
        # prep.differentGray()
        prep.gray()
        # prep.differentGray2()
        prep.morphologicalOpen()
        # prep.adaptiveThreshold(23, 7)
        # prep.division(95)
        # prep.morphologicalOpen()

        prep.otsu()
        # prep.applyContrast(1.5, -50)
        # prep.applyBlur(5)
        # prep.pyrMeanShiftFilterContours()
        # prep.applyContrast(1.25, -25)
        # prep.pyrMeanShiftFilterContours()
        # prep.applyContrast(1.25, -25)
        # prep.pyrMeanShiftFilterContours()
        # prep.applyContrast(1.25, 0)
        # prep.removeShadow()
        # prep.getSaturation()

        copyImage = prep.getImage()
        contours = shapeFinder.detectShapes3D(copyImage, originalImage)

    else:
        if jigsaw:
            # GIVE ME JIGSAW CONTOURS.
            prep.jigsaw2D()
            copyImage = prep.getImage()
            contours = shapeFinder.detectJigsaw(copyImage, originalImage)
        else:
            prep.basic2D()

            copyImage = prep.getImage()
            contours = shapeFinder.detectShapes2D(copyImage, originalImage)

    # Rotate the images in the case they are at an angle.
    rotatedContours = contours
    if not jigsaw:
        rotator = Rotator(rotatorLog | allLog)
        rotatedContours = rotator.rotate(contours, originalImage)

    # Find the units in which to break the shapes.
    # Grid the smallest rectangle in a grid with units lxl, where l is a divisor of the smallest side.
    # Look for the biggest l s.t. the area lost in the process is less than a given percent.
    processor = Processor(rotatedContours, processorLog | allLog, jigsaw)
    if jigsaw:
        processor.findGridsJigsaw(rows, columns)
    else:
        processor.findGrids()
    pieces = processor.getPieces()

    if show:
        print("Here are the grids for the pieces:")
        print(pieces)

    colour = args["colour"]
    cpp = args["cpp"]

    bkt = -1
    if args["BKT"] is not None:
        bkt = int(args["BKT"])

    dlx = -1
    if args["DLX"] is not None:
        dlx = int(args["DLX"])

    nonScale = args["noScaling"]
    puzzleSolver = Solver(solverLog | allLog, originalImage, bkt, dlx, colour, cpp, jigsaw, nonScale)

    if puzzleSolver.solveBackTracking(pieces):
        if show:
            if not jigsaw:
                rgbArray = np.array(puzzleSolver.getSolution()).astype(np.uint8)

                # Display the RGB array using Matplotlib
                plt.imshow(rgbArray)
                plt.axis('off')  # Turn off axis labels
                plt.show()
            else:
                # print("Check this: ", puzzleSolver.getAllSolutions())
                boardP = puzzleSolver.getBoardPiece()
                cv.imwrite("boardP.png", boardP.getOriginalContour().getImage())
                boardImg = boardP.getOriginalContour().getImage()
                # targetHash = computeHash(boardImg)
                h, w = boardImg.shape[:2]
                solutions = puzzleSolver.getAllSolutions()
                drawnSolutions = []
                timeBeforePrint = timer()
                for idx in range(len(solutions)):
                    currSol = printJigsaw(solutions[idx], puzzleSolver.getDictPieces(), originalImage, puzzleSolver.getColourMap(), w, h, idx)
                    drawnSolutions.append(currSol)
                timeAfterPrint = timer()
                timeTookPrint = timeAfterPrint - timeBeforePrint
                print("time took printing: ", timeTookPrint)
                # hashes = computeAllHashes(drawnSolutions)
                # print("no way?: ", hashes)
                #
                # indexBest, d = findBestSolutionWithHashes(hashes, targetHash)
                # print("which one?: ", indexBest)
                # timeInSSIM = timer()
                # bestImg, ssim = findBestSolutionSSIM(drawnSolutions, boardImg)
                # timeOutSSIM = timer()
                # timeSSIM = timeOutSSIM - timeInSSIM
                print("Got here!")
                timeInNCC = timer()
                bestImg, ncc = findBestSolutionNCC(drawnSolutions, boardImg)
                timeOutNCC = timer()
                timeNCC = timeOutNCC - timeInNCC

                # print("SSIM: ", timeSSIM)
                print("NCC: ", timeNCC)
                # cv.imwrite("iazima.png", drawnSolutions[indexBest])
                cv.imwrite("iazima.png", bestImg)
                printJigsaw(puzzleSolver.getOutput(), puzzleSolver.getDictPieces(), originalImage, puzzleSolver.getColourMap(), w, h,0)
                rgbArray = np.array(puzzleSolver.getSolution()).astype(np.uint8)

                # Display the RGB array using Matplotlib
                plt.imshow(rgbArray)
                plt.axis('off')  # Turn off axis labels
                plt.show()

        print("Puzzle is solvable.")
    else:
        print("Something is or went wrong with the puzzle.")


if __name__ == '__main__':
    main()
