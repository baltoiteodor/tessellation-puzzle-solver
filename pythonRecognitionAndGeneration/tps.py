import argparse
import json
import os
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # Load original image.
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

    #
    ## Starting timer for the whole process.
    #

    timeStartProject = timer()
    possibleSols = 0


    prep = PreProcessor(copyImage)
    if realProc:

        #   Testing thresh:
        # prep.applyContrast(1.5, 0)
        # prep.removeShadow()
        prep.morphologicalOpen()

        # prep.applyBlur(53)
        prep.applyContrast(2.0, -50)

        # prep.hueChannel()
        # prep.lab()
        # prep.bilateralFilter()
        # prep.morphologicalOpen()
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
            # Preprocess images of Jigsaw puzzles and obtain their contours.
            prep.jigsaw2DV2()
            copyImage = prep.getImage()
            contours = shapeFinder.detectJigsaw(copyImage, originalImage)
        else:
            # Preprocess images of traditional tessellation puzzles and obtain their contours.
            prep.basic2D()
            copyImage = prep.getImage()
            contours = shapeFinder.detectShapes2D(copyImage, originalImage)

    # Rotate the images in the case they are at an angle.
    rotatedContours = contours
    rotator = Rotator(rotatorLog | allLog)
    if not jigsaw:
        if realProc:
            # Room for improving.
            rotatedContours = rotator.rotate(contours, originalImage)
        else:
            rotatedContours = rotator.rotate(contours, originalImage)

    # Find the unit length and process the contours into grids.
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

    # Instantiate the Solver with the corresponding options enabled.
    puzzleSolver = Solver(solverLog | allLog, originalImage, bkt, dlx, colour, cpp, jigsaw, nonScale)

    chooserTimeTotal = 0
    # We set this to False in the case the algorithm fails to find a suitable number of potential solutions.
    jigsawSuccess = True
    ncc = 0
    if puzzleSolver.solveBackTracking(pieces):
        if show:
            if not jigsaw:
                rgbArray = np.array(puzzleSolver.getSolution()).astype(np.uint8)

                # Original way of printing the puzzle result is commented out.

                # Display the RGB array using Matplotlib
                # plt.imshow(rgbArray)
                # plt.axis('off')  # Turn off axis labels
                # plt.show()
                # print(puzzleSolver.getOutput())

                # Print a puzzle with outlines for each piece.
                printTessellation(puzzleSolver.getOutput(), puzzleSolver.getDictPieces())

        if jigsaw:
            # Solving jigsaw puzzles.
            boardP = puzzleSolver.getBoardPiece()
            boardImg = boardP.getOriginalContour().getImage()
            h, w = boardImg.shape[:2]
            solutions = puzzleSolver.getAllSolutions()

            if len(solutions) == 0:
                jigsawSuccess = False

            possibleSols = len(solutions)
            colourDictionary = puzzleSolver.getColourDict()

            chooserTimeIn = timer()

            # Uncomment these to enable sequentially choosing the best solution.

            # drawnSolutions = []
            # timeBeforePrint = timer()

            # for idx in range(len(solutions)):
            #     currSol, timeTk, timfff, incr = printJigsawOptimised(solutions[idx], puzzleSolver.getDictPieces(), originalImage, puzzleSolver.getColourMap(), w, h, idx, rows, columns, colourDictionary)
                # drawnSolutions.append(currSol)

            # timeAfterPrint = timer()
            # timeTookPrint = timeAfterPrint - timeBeforePrint
            # print("time took printing with no parallelism: ", timeTookPrint)

            def generateSolution(idx):
                return printJigsawOptimised(solutions[idx], puzzleSolver.getDictPieces(), originalImage, puzzleSolver.getColourMap(), w, h, idx, rows, columns, colourDictionary)

            # Parallelised choosing of best option.
            drawnSolutionsParallel = []
            timeBeforePrintParallel = timer()
            with ThreadPoolExecutor() as executor:
                futureToIdx = {executor.submit(generateSolution, idx): idx for idx in range(len(solutions))}
                for future in as_completed(futureToIdx):
                    drawnSolutionsParallel.append(future.result())
            timeAfterPrintParallel = timer()
            timeTookPrintParallel = timeAfterPrintParallel - timeBeforePrintParallel

            # Uncomment to inspect the time taken to recreate all solutions.
            # print(f"Time taken with parallelism: {timeTookPrintParallel} seconds")

            # Uncomment to choose the best solution with pHash or SSIM.

            # hashes = computeAllHashes(drawnSolutions)
            # indexBest, d = findBestSolutionWithHashes(hashes, targetHash)
            # timeInSSIM = timer()
            # bestImg, ssim = findBestSolutionSSIM(drawnSolutionsParallel, boardImg)
            # timeOutSSIM = timer()
            # timeSSIM = timeOutSSIM - timeInSSIM
            # print("SSIM: ", timeSSIM)

            timeInNCC = timer()
            bestImg, ncc = findBestSolutionNCC(drawnSolutionsParallel, boardImg)
            timeOutNCC = timer()
            timeNCC = timeOutNCC - timeInNCC

            # print("NCC: ", timeNCC)
            ncc = timeNCC
            chooserTimeOut = timer()


            chooserTimeTotal = chooserTimeOut - chooserTimeIn

            cv.imwrite("jigsawSolution.png", bestImg)

            # Tease a rough sketch of the solution.
            rgbArray = np.array(puzzleSolver.getSolution()).astype(np.uint8)
            if show:
                plt.imshow(rgbArray)
                plt.axis('off')  # Turn off axis labels
                plt.show()

        print("Puzzle is solvable.")
    else:
        jigsawSuccess = False
        print("Something is or went wrong with the puzzle.")


    timeStopProject = timer()
    timeTakenProject = timeStopProject - timeStartProject

    #
    ## Print statistics about times taken in each step of the system in a corresponding file.
    #

    dir = 'Evaluation'

    baseName = os.path.splitext(os.path.basename(args['image']))[0]

    fileName = baseName + '.txt'

    # Place the statistics in the corresponding subdirectory based on the type of algorithm used by the Solver.
    if bkt >= 0:
        dir = os.path.join(dir, 'BKT')
    else:
        if cpp:
            dir = os.path.join(dir, 'DLX-CPP')
        else:
            dir = os.path.join(dir, 'DLX')

    # Further split into subdirectories based on the colour option.
    if colour:
        dir = os.path.join(dir, 'Colour')
    else:
        dir = os.path.join(dir, 'No-Colour')

    outputFile = os.path.join(dir, fileName)

    # Retrieve the number of pieces of the puzzle based on the naming convention. (e.g. cat35.png)
    numPieces = len(puzzleSolver.getDictPieces().keys())

    # Add size of the image to the statistics.
    success, encodedImage = cv2.imencode('.png', originalImage)
    imageSizeKB = 0
    if success:
        # Get the size of the encoded image in bytes first.
        imageSizeBytes = len(encodedImage.tobytes())
        # Convert the size to KB.
        imageSizeKB = imageSizeBytes / 1024

    # Put together all stats.
    timesTaken = [
        {"label": "PreProcessing", "time": prep.getTimeTaken()},
        {"label": "Contour Finding", "time": shapeFinder.getTimeTaken()},
        {"label": "Rotating Pieces", "time": rotator.getTimeTaken()},
        {"label": "Processing Pieces into Grids", "time": processor.getTimeTaken()},
        {"label": "Solving the Puzzle", "time": puzzleSolver.getTimeTaken()},
        {"label": "First BKT solution found at ", "time": puzzleSolver.getFirstSolTime()},
        {"label": "Choosing the best solution for Jigsaw", "time": chooserTimeTotal},
        {"label": "Total time", "time": timeTakenProject},
        {"label": "Jigsaw Success", "time": jigsawSuccess},
        {"label": "Pieces", "time": numPieces},
        {"label": "NCC", "time": ncc},
        {"label": "Possible Solutions", "time": possibleSols},
        {"label": "Image size", "time": imageSizeKB}

    ]

    # Write the statistics to the file.
    try:
        with open(outputFile, 'w') as file:
            file.write("Label\tTime\n")
            for entry in timesTaken:
                file.write(f"{entry['label']}\t{entry['time']}\n")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

if __name__ == '__main__':
    main()
