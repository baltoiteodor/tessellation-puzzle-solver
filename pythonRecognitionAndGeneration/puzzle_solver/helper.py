import cv2
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000

from PIL import Image, ImageDraw
from misc.piece import *

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

from timeit import default_timer as timer

def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)

COLOURTHRESHOLD = 20


def resizeToDimensions(image, contour, targetWidth, targetHeight):
    # Convert the contour to a numpy array if it isn't already.
    contournp = np.array(contour, dtype=np.int32)

    # Calculate the bounding box of the contour.
    x, y, w, h = cv2.boundingRect(contournp)

    # Calculate separate scale factors for width and height.
    widthScale = targetWidth / w
    heightScale = targetHeight / h

    # Scale the contour by applying different scale factors to x and y coordinates.
    scaledContour = np.column_stack((
        (contournp[:, :, 0] - x) * widthScale,
        (contournp[:, :, 1] - y) * heightScale
    )).astype(np.int32)

    # Extract the region of interest (ROI) from original image.
    roi = image[y:y + h, x:x + w]

    # Uncomment to print original contour.
    # cv2.imwrite('Notscaled.png', roi)

    # Resize the ROI to the target dimensions.
    scaledROI = cv2.resize(roi, (targetWidth, targetHeight), interpolation=cv2.INTER_LINEAR)

    # Create a mask for the scaled contour.
    scaledMask = np.zeros((targetHeight, targetWidth), dtype=np.uint8)
    cv2.drawContours(scaledMask, [scaledContour], -1, (255), thickness=cv2.FILLED)

    # Apply the scaled mask to the scaled ROI.
    scaledROIWithMask = cv2.bitwise_and(scaledROI, scaledROI, mask=scaledMask)

    # Uncomment this if you want to inspect the scaled contour.
    # cv2.imwrite('scaled.png', scaledROIWithMask)

    return scaledContour, scaledROIWithMask


def scalePiece(piece: Piece, scaleFactor, image):
    originalContour = piece.getOriginalContour().getContour()

    originalContournp = np.array(originalContour, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(originalContournp)

    # Resize the bounding box dimensions.
    scaledW = int(w * scaleFactor)
    scaledH = int(h * scaleFactor)

    # Scale and translate the contour.
    scaledContour = ((originalContournp - (x, y)) * scaleFactor).astype(np.int32)

    # Create a mask for the contour.
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [scaledContour], -1, (255), thickness=cv2.FILLED)

    # Extract the region of interest (ROI) from the original image.
    roi = image[y:y + h, x:x + w]

    # Resize the ROI using the scaled dimensions.
    scaledRoi = cv2.resize(roi, (scaledW, scaledH), interpolation=cv2.INTER_LINEAR)

    # Create a mask for the scaled contour.
    scaledMask = np.zeros_like(scaledRoi[:, :, 0])
    cv2.drawContours(scaledMask, [scaledContour], -1, (255), thickness=cv2.FILLED)

    # Apply the scaled mask to the scaled ROI.
    scaledRoiWithMask = cv2.bitwise_and(scaledRoi, scaledRoi, mask=scaledMask)

    # Uncomment to inspect the scaled version.
    # cv2.imwrite('scaled.png', scaledRoiWithMask)

    # Transform into a Contour object with new image.
    newContour = Contour(scaledContour, scaledRoiWithMask, piece.orderNum())
    unitLen = piece.getUnitLen()

    # Create new grid. If the error is too great return failure to scale the board.
    # Could be abstracted from the processor class.
    coveredArea = 0.0
    pieceArea = newContour.getArea()

    box = cv2.boxPoints(newContour.getMinAreaRect())
    box = np.int0(box)
    # x is width, y is height.
    # Obtain the corners of the rectangle.
    topLeftX = np.min(box[:, 0])
    topLeftY = np.min(box[:, 1])
    botRightX = np.max(box[:, 0])
    botRightY = np.max(box[:, 1])

    # For each unit of the grid, check if the centre is inside the polygon, if yes, then put 1 inside the
    # grid, otherwise 0.
    # Start with row 0, stop when we are outside the rectangle. Same for columns.
    unitX = topLeftX
    indexX = 0

    # Due to width/height switching I will calculate my own.
    width = botRightX - topLeftX
    height = botRightY - topLeftY
    rows = int(width / unitLen + 1)
    cols = int(height / unitLen + 1)

    # Invert the x, y to y, x in the grid, so it looks like in the image.
    # Initiate the grid.
    grid = np.zeros((cols, rows))

    # Use this to determine if the piece is rotatable.
    noOnes: int = 0
    while unitX < botRightX:  # When the new unit x coordinate is out of bounds.
        indexY = 0
        unitY = topLeftY
        # Loop columns.
        while unitY < botRightY:
            # Find centre of grid unit, check if inside the contour.
            centreUnit = (int(unitX + unitLen / 2), int(unitY + unitLen / 2))
            isIn = cv2.pointPolygonTest(newContour.getContour(), centreUnit, False)

            if isIn >= 0:
                # Mark this unit as 1 in the grid.
                grid[indexY][indexX] = 1
                noOnes += 1
            else:
                grid[indexY][indexX] = 0

            # Add to covered area
            coveredArea += grid[indexY][indexX] * unitLen * unitLen
            unitY += unitLen
            indexY += 1
        unitX += unitLen
        indexX += 1

    grid = grid.astype(int)

    # Remove borderline zeroes.
    grid = trimGrid(grid)

    newPiece: Piece = Piece(newContour, grid, newContour.getColour(), unitLen, (topLeftX, topLeftY))

    # Error is the maximum error per piece.
    error = abs(1 - coveredArea / pieceArea)

    if error > 0.05:
        return False, None

    return True, newPiece


# Rounds to closest 0.05.
def roundScaler(scaler):
    return round(scaler * 20) / 20


def calculatePiecesArea(pieces: Pieces):
    area = 0.0
    for pc in pieces:
        area += pc.getOriginalContour().getArea()
    return area


def findClosestContourPoint(contour, point):

    # Reshape contour if it is in the wrong format.
    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour.reshape(-1, 1, 2)
    elif contour.ndim != 3:
        raise ValueError("Contour must be a 3-dimensional array with shape (n, 1, 2).")

    # Initialize minimum distance and the closest point.
    minDist = float('inf')
    closestPoint = None

    # Iterate over each point in the contour.
    for contourPoint in contour:
        # Calculate the Euclidean distance between the given point and the current contour point.
        dist = np.linalg.norm(contourPoint[0] - point)

        # Update minimum distance and closest point if current distance is smaller.
        if dist < minDist:
            minDist = dist
            closestPoint = tuple(contourPoint[0])

    return closestPoint


def findClosestPointOnMask(mask, target_point):
    # Get indices where the mask is non-zero (the contour).
    ys, xs = np.nonzero(mask)
    contourPoints = np.column_stack((xs, ys))

    # Calculate distances.
    distances = np.sqrt((contourPoints[:, 0] - target_point[0]) ** 2 + (contourPoints[:, 1] - target_point[1]) ** 2)

    # Get closest point.
    closestPointIdx = np.argmin(distances)
    closestPoint = contourPoints[closestPointIdx]

    return tuple(closestPoint)


# Get the pieceID piece out of the outputMatrix and its corresponding colourMap.
def retrieveShape(outputMatrix, pieceId, row, col, colourMap):
    shape = np.zeros_like(outputMatrix)
    pieceColours = [[(0, 0, 0) for _ in range(len(outputMatrix[0]))] for _ in range(len(outputMatrix))]
    fill(outputMatrix, pieceId, row, col, shape, len(outputMatrix), len(outputMatrix[0]), pieceColours, colourMap)
    shape = trimGrid(shape)
    colours = trimColourGrid(pieceColours)
    return shape, colours


def fill(outputMatrix, pieceId, row, col, shape, lenRows, lenCols, pieceColours, colourMap):
    if outputMatrix[row][col] == pieceId:
        shape[row][col] = 1
        pieceColours[row][col] = colourMap[row][col]
        outputMatrix[row][col] = 0
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for idx in range(8):
            rowx = row + dx[idx]
            colx = col + dy[idx]
            if insideMatrix(rowx, colx, lenRows, lenCols):
                fill(outputMatrix, pieceId, rowx, colx, shape, lenRows, lenCols, pieceColours, colourMap)


def insideMatrix(r, c, n, m):
    return 0 <= r < n and 0 <= c < m


# Used this previously to recreate jigsaw solutions.
# def printJigsawBad(outputMatrix, dictToPieces, originalImage):
#     # Map pieces to their top left corner, as well as establish a relation between the corner and the top left of the
#     # piece bounding rectangle.
#     dictToLeftCorners = {}
#     dictToRightCorners = {}
#     dictToMoveVectorsRect = {}
#     # Distance from right corner of jigsaw to left corner of jigsaw.
#     dictToMoveVectorsPiece = {}
#
#     for idx in dictToPieces.keys():
#         currentPiece = dictToPieces[idx]
#         x, y, w, h = currentPiece.getOriginalContour().getBoundingRect()
#         topLeftRect = (x, y)
#
#         topRightRect = (x + w, y)
#         botLeftRect = (x, y + h)
#         botRightRect = (x + w, y + h)
#         topLeftCorner = findClosestContourPoint(currentPiece.getOriginalContour().getContour(), np.array(topLeftRect))
#         topRightCorner = findClosestContourPoint(currentPiece.getOriginalContour().getContour(), np.array(topRightRect))
#         moveVectorRect = np.array(topLeftCorner) - np.array(topLeftRect)
#         moveVectorPiece = np.array(topRightCorner) - np.array(topLeftCorner)
#
#         # if moveVectorPiece[1] > 0:
#         #
#         #     img = np.zeros((2000, 2000, 3), dtype=np.uint8)
#         #
#         #     # Draw the contour
#         #     cv2.drawContours(img, [currentPiece.getOriginalContour().getContour() - np.array([[600, 1200]])], -1, (0, 255, 0), 2)  # Green for the contour
#         #
#         #     # Draw circles for the corners
#         #     cv2.circle(img, topLeftCorner - np.array((600, 1200)), 5, (255, 0, 0), -1)  # Red for topLeftCorner
#         #     cv2.circle(img, topRightCorner - np.array((600, 1200)), 5, (0, 0, 255), -1)  # Blue for topRightCorner
#         #     window_name = f"Contour {idx}"
#         #     cv2.imshow(window_name, img)
#         #     cv2.waitKey(0)
#
#         # print("helo here: ", moveVectorPiece, moveVectorRect)
#         dictToLeftCorners[idx] = topLeftCorner
#         dictToRightCorners[idx] = topRightCorner
#         dictToMoveVectorsRect[idx] = moveVectorRect
#         dictToMoveVectorsPiece[idx] = moveVectorPiece
#
#     # Starting to build the image.
#     # print("Hmmmm: ", originalImage.shape)
#     solutionImage = np.zeros(originalImage.shape, dtype=np.uint8)
#     nextTopLeft = np.array((0, 0))
#     # Will use this dictionary to mark what pieces were already placed when iterating through the outputMatrix.
#     piecesDone = {}
#     piecesDone[0] = True
#     for row in range(len(outputMatrix)):
#         for col in range(len(outputMatrix[row])):
#             if not (outputMatrix[row][col] in piecesDone):
#                 if row > 0:
#                     cv2.imwrite("progress.png", solutionImage)
#                     return
#                 pieceId = outputMatrix[row][col]
#                 # Fill shape of piece in the output matrix.
#                 pieceShape = retrieveShape(outputMatrix, pieceId, row, col)
#                 # print("Here is a piece shape extracted: ", pieceShape)
#                 currentPiece = dictToPieces[pieceId]
#                 angle = currentPiece.retrieveAngle(pieceShape)
#                 # print("Angle: ", angle)
#                 # print("Current piece and stuff: ", pieceId, row, col, nextTopLeft)
#                 # print(dictToPieces[pieceId])
#                 piecesDone[pieceId] = True
#                 # corner = dictToCorners[pieceId]
#                 targetLocation = nextTopLeft - dictToMoveVectorsRect[pieceId]
#                 # print("Target Location: ", targetLocation)
#                 currContour = currentPiece.getOriginalContour()
#                 currContour.createROIAtAngle(targetLocation, solutionImage, angle)
#                 # Calculate the new nextTopLeft somehow. Might be top right of the piece, aka the closest point
#                 # to the top right of the bounding rectangle. Add move vector from the 2 corners to the nextTopLeft probably?
#                 nextTopLeft += dictToMoveVectorsPiece[pieceId]
#                 cv2.imwrite("progress.png", solutionImage)
#
#     cv2.imwrite("progress.png", solutionImage)


def rotateImage(image, angle, center):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    newWidth = int((height * sin) + (width * cos))
    newHeight = int((height * cos) + (width * sin))
    matrix[0, 2] += (newWidth / 2) - center[0]
    matrix[1, 2] += (newHeight / 2) - center[1]
    rotatedImg = cv2.warpAffine(image, matrix, (newWidth, newHeight))

    # Uncomment this to inspect the rotated image.
    # cv2.imwrite(f"rotated{angle}.png", rotatedImg)

    return rotatedImg

# Draw the piece based on the nextTopLeft of the current solution image and angle.
def placeAndRotateContour(contour, angle, originalImg, targetImg, nextTopLeft, unit):
    # Create a mask and extract the ROI.
    mask = np.zeros(originalImg.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    roi = cv2.bitwise_and(originalImg, originalImg, mask=mask)
    extracted = roi[y:y + h, x:x + w]
    maskExtracted = mask[y:y + h, x:x + w]

    # Uncomment to inspect the extracted mask.
    # cv2.imwrite(f"maskPrior{angle}.png", maskExtracted)

    # Rotate the contour based on the angle.
    center = (w // 2, h // 2)
    rotatedImg = rotateImage(extracted, angle, center)
    rotatedMask = rotateImage(maskExtracted, angle, center)

    # Uncomment to inspect the rotated mask.
    # cv2.imwrite(f"ma{angle}.png", rotatedMask)

    # Get the corners of the rotated mask.
    nonZeroPoints = np.column_stack(np.where(rotatedMask > 0))
    minX, minY = nonZeroPoints.min(axis=0)
    maxX, maxY = nonZeroPoints.max(axis=0)

    topLeftRect = (minY, minX)

    # topRightRect = (maxY, minX)
    # botLeftRect = (minY, maxX)

    # timeIn = timer()
    topLeftCorner = findClosestPointOnMask(rotatedMask, np.array(topLeftRect))
    # topRightCorner = findClosestPointOnMask(rotatedMask, np.array(topRightRect))
    # botLeftCorner = findClosestPointOnMask(rotatedMask, np.array(botLeftRect))

    topRightCorner = (topLeftCorner[0] + 3 * unit, topLeftCorner[1])
    botLeftCorner = (topLeftCorner[0], topLeftCorner[1] + 3 * unit)
    # timeOut = timer()

    # Uncomment these for some stats.
    # print("Times: ", timeOut - timeIn)
    # print("Current target: ", nextTopLeft)
    # print("offset", topLeftRect[0] - topLeftCorner[0], topLeftRect[1] - topLeftCorner[1])
    # print("Bounding: ", botLeftRect, topLeftRect, topRightRect)
    # print(topLeftCorner, topRightCorner)

    # Take into account that the thumbs have irregular sizes.
    placementX = nextTopLeft[0] + (topLeftRect[0] - topLeftCorner[0])
    placementY = nextTopLeft[1] + (topLeftRect[1] - topLeftCorner[1])

    placementX = max(placementX, 0)
    placementY = max(placementY, 0)

    # Where the piece will be placed.
    targetArea = targetImg[placementY:placementY + rotatedImg.shape[0],
                  placementX:placementX + rotatedImg.shape[1]]

    maskResized = cv2.resize(rotatedMask, (targetArea.shape[1], targetArea.shape[0]))

    targetRegionMasked = cv2.bitwise_and(targetArea, targetArea, mask=cv2.bitwise_not(maskResized))
    finalRegion = cv2.add(targetRegionMasked, cv2.bitwise_and(rotatedImg, rotatedImg, mask=maskResized))

    # Add the piece to the current solution image.
    targetImg[placementY:placementY + rotatedImg.shape[0],
    placementX:placementX + rotatedImg.shape[1]] = finalRegion

    # For printing when unit length is not taken in consideration.
    newNextTopLeft = (
        nextTopLeft[0] + (topRightCorner[0] - topLeftCorner[0]),
        nextTopLeft[1] + (topRightCorner[1] - topLeftCorner[1]))
    newNextBotLeft = (
        nextTopLeft[0] + (botLeftCorner[0] - topLeftCorner[0]), nextTopLeft[1] + (botLeftCorner[1] - topLeftCorner[1]))

    return newNextTopLeft, newNextBotLeft


def getNextRow(outputMatrix, row):
    for r in range(row, len(outputMatrix)):
        if outputMatrix[r][0] != 0:
            return r
    return -1

def placeAndRotateContourOptimised(contourObject, angle, originalImg, targetImg, nextTopLeft, unit):
    # Create a mask and extract the ROI.
    mask = np.zeros(originalImg.shape[:2], dtype=np.uint8)
    contour = contourObject.getContour()
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = contourObject.getBoundingRect()
    roi = cv2.bitwise_and(originalImg, originalImg, mask=mask)
    extracted = roi[y:y + h, x:x + w]
    maskExtracted = mask[y:y + h, x:x + w]

    # Uncomment to inspect the extracted mask.
    # cv2.imwrite(f"maPrior{angle}.png", maskExtracted)

    center = (w // 2, h // 2)
    rotatedImg = rotateImage(extracted, angle, center)
    rotatedMask = rotateImage(maskExtracted, angle, center)

    # Uncomment to inspect the rotated mask.
    # cv2.imwrite(f"ma{angle}.png", rotatedMask)

    # Get the corners of the rotated mask.
    nonZeroPoints = np.column_stack(np.where(rotatedMask > 0))
    minX, minY = nonZeroPoints.min(axis=0)

    topLeftRect = (minY, minX)
    topLeftCorner = findClosestPointOnMask(rotatedMask, np.array(topLeftRect))

    # Take into account that the thumbs have irregular sizes.
    placementX = nextTopLeft[0] + (topLeftRect[0] - topLeftCorner[0])
    placementY = nextTopLeft[1] + (topLeftRect[1] - topLeftCorner[1])

    placementX = max(placementX, 0)
    placementY = max(placementY, 0)

    targetArea = targetImg[placementY:placementY + rotatedImg.shape[0],
                  placementX:placementX + rotatedImg.shape[1]]

    maskResized = cv2.resize(rotatedMask, (targetArea.shape[1], targetArea.shape[0]))

    targetRegionMasked = cv2.bitwise_and(targetArea, targetArea, mask=cv2.bitwise_not(maskResized))
    finalRegion = cv2.add(targetRegionMasked, cv2.bitwise_and(rotatedImg, rotatedImg, mask=maskResized))

    # Add the piece to the current solution image.
    targetImg[placementY:placementY + rotatedImg.shape[0],
    placementX:placementX + rotatedImg.shape[1]] = finalRegion

    return
def printJigsawOptimised(outputMatrix, dictToPieces, originalImage, colourMap, w, h, idx, rows, cols, colourDict):
    solutionImage = np.zeros(originalImage.shape, dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            pieceId = outputMatrix[3 * row][3 * col]
            # Fill shape of piece in the output matrix.
            pieceShape, pieceColourGrid = retrieveShape(outputMatrix, pieceId, 3 * row, 3 * col, colourMap)
            currentPiece = dictToPieces[pieceId]
            unit = currentPiece.getUnitLen()
            nextTopLeft = (col * 3 * unit, row * 3 * unit)
            angle = currentPiece.retrieveAngle(pieceShape, pieceColourGrid, colourDict)
            # Get the piece contour and rotate it at the desired angle.
            currContour = currentPiece.getOriginalContour()
            placeAndRotateContourOptimised(currContour, 360 - angle,
                                    currentPiece.getOriginalContour().getImage(),
                                    solutionImage, nextTopLeft, unit)
    solutionImage = solutionImage[:h, :w]
    return solutionImage

def printJigsaw(outputMatrix, dictToPieces, originalImage, colourMap, w, h, idx, rows, cols, colourDict):
    solutionImage = np.zeros(originalImage.shape, dtype=np.uint8)
    nextTopLeft = np.array((0, 0))
    piecesDone = {}
    piecesDone[0] = True
    # In our case, we will have the same amount of pieces on each row, so we can store the botLefts and
    # keep them for future topLefts.
    botLefts = {}
    beenThru = 0
    for row in range(len(outputMatrix)):
        row = getNextRow(outputMatrix, row)
        currIndex = 0
        if row == -1:
            break
        for col in range(len(outputMatrix[row])):
            if not (outputMatrix[row][col] in piecesDone):
                pieceId = outputMatrix[row][col]
                # Fill shape of piece in the output matrix.
                pieceShape, pieceColourGrid = retrieveShape(outputMatrix, pieceId, row, col, colourMap)
                currentPiece = dictToPieces[pieceId]
                unit = currentPiece.getUnitLen()
                angle = currentPiece.retrieveAngle(pieceShape, pieceColourGrid, colourDict)
                piecesDone[pieceId] = True

                # Get the piece contour and rotate it at the desired angle.
                currContour = currentPiece.getOriginalContour().getContour()
                if beenThru == 0:
                    nextTopLeft, botLeft = placeAndRotateContour(currContour, 360 - angle,
                                                                 currentPiece.getOriginalContour().getImage(),
                                                                 solutionImage, nextTopLeft, unit)
                else:
                    nextTopLeft, botLeft = placeAndRotateContour(currContour, 360 - angle,
                                                                 currentPiece.getOriginalContour().getImage(),
                                                                 solutionImage, botLefts[currIndex], unit)
                botLefts[currIndex] = botLeft
                currIndex += 1
        beenThru += 1

    solutionImage = solutionImage[:h, :w]

    # Uncomment to see progress.
    # cv2.imwrite(f"progress{idx}.png", solutionImage)
    return solutionImage


def computeHash(image):
    image = Image.fromarray(image)
    return imagehash.phash(image)

def computeAllHashes(images):
    def computeHashF(image):
        return computeHash(image)

    hashes = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(computeHashF, img) for img in images]
        for future in as_completed(futures):
            hashValue = future.result()
            hashes.append(hashValue)

    return hashes

# pHash for choosing the best solution.
def findBestSolutionWithHashes(hashes, targetHash):
    minDistance = float('inf')
    mostSimilarIndex = None

    for index, solutionHash in enumerate(hashes):
        distance = targetHash - solutionHash
        # print("potential: ", distance)
        if distance < minDistance:
            minDistance = distance
            mostSimilarIndex = index

    return mostSimilarIndex, minDistance


def computeSSIM(image1, image2):
    grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayImage1, grayImage2, full=True)
    return score


def computeNCC(image1, image2):
    grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(grayImage1, grayImage2, cv2.TM_CCORR_NORMED)
    _, maxVal, _, _ = cv2.minMaxLoc(result)
    return maxVal

# Choose a solution based on NCC score.
def findBestSolutionNCC(potentialSolutions, targetImage):
    maxNCC = -float('inf')
    mostSimilarImage = None

    def computeImageNCC(solutionImage):
        return computeNCC(solutionImage, targetImage)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(computeImageNCC, img): img for img in potentialSolutions}
        for future in as_completed(futures):
            nccScore = future.result()
            solutionImage = futures[future]
            # if nccScore > 0.98:
            #     return solutionImage, nccScore
            if nccScore > maxNCC:
                maxNCC = nccScore
                mostSimilarImage = solutionImage

    return mostSimilarImage, maxNCC

# Choose solution based on SSIM score.
def findBestSolutionSSIM(potentialSolutions, targetImage):
    maxSSIM = -float('inf')
    mostSimilarImage = None

    def computeImageSSIM(solutionImage):
        return computeSSIM(solutionImage, targetImage)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(computeImageSSIM, img): img for img in potentialSolutions}
        for future in as_completed(futures):
            ssimScore = future.result()
            solutionImage = futures[future]
            if ssimScore > maxSSIM:
                maxSSIM = ssimScore
                mostSimilarImage = solutionImage

    return mostSimilarImage, maxSSIM


# No concurrency image differentiation. Used for comparing times to the concurrent version.
def findMostSimilarImage(potentialSolutions, targetImage):
    maxSSIM = -float('inf')
    mostSimilarImage = None

    for solution_image in potentialSolutions:
        ssimScore = compute_ssim(solution_image, targetImage)
        if ssimScore > maxSSIM:
            maxSSIM = ssimScore
            mostSimilarImage = solution_image

    return mostSimilarImage, maxSSIM


def trimGrid(grid):
    # Remove the last columns if all zero.
    while grid.shape[1] > 0 and np.all(grid[:, -1] == 0):
        grid = grid[:, :-1]
    # Remove leading columns with all zeros
    while grid.shape[1] > 0 and np.all(grid[:, 0] == 0):
        grid = grid[:, 1:]
    # Remove the last row if all zero.
    while grid.shape[0] > 0 and np.all(grid[-1, :] == 0):
        grid = grid[:-1, :]
    # Remove leading rows with all zeros
    while grid.shape[0] > 0 and np.all(grid[0, :] == 0):
        grid = grid[1:, :]

    return grid


def findBoard(pieces: Pieces):
    board = None
    maxSize = 0
    boardIndex = -1

    for index, piece in enumerate(pieces):
        # Check only pieces that are not rotatable (a.k.a. pieces containing only 1s).
        if piece.isBoardable():
            currentSize = piece.area()

            if currentSize > maxSize:
                maxSize = currentSize
                board = piece
                boardIndex = index

    if maxSize > 0:
        del pieces[boardIndex]
    return board


def removePiece(currBoard: Board, outputMatrix: Board, piece: Piece, row: int, col: int):
    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            currBoard[i][j] -= piece.pixelAt(i - row, j - col)
            if piece.pixelAt(i - row, j - col):
                outputMatrix[i][j] = 0


def setPiece(currBoard: Board, board: Board, outputMatrix: Board,
             piece: Piece, row: int, col: int):
    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            if currBoard[i][j] == 0:
                outputMatrix[i][j] = piece.orderNum()
            currBoard[i][j] += piece.pixelAt(i - row, j - col)


def rotatePiece(piece: Piece):
    piece.rotatePiece()


def rotatePieceNonOptimal(piece: Piece):
    rotatedGrid = np.zeros((piece.columns(), piece.rows()), dtype=int)
    for i in range(piece.rows()):
        for j in range(piece.columns()):
            rotatedGrid[j][piece.rows() - i - 1] = piece.pixelAt(i, j)

    piece.setGrid(rotatedGrid)
    oldRows = piece.rows()
    oldCols = piece.columns()
    piece.setRowsNum(oldCols)
    piece.setColsNum(oldRows)
    piece.increaseCurrentRotation()


# Returns True if the piece fits in nicely, otherwise False.
def isValid(currBoard: Board, targetBoard: Board, colourMap, piece: Piece, row: int, col: int, colourMatters: bool, dict):
    # Pieces will have leading 0s in the matrix like the + sign. In this case, change the row, piece of where to put
    # the piece by the leading amount of 0s on the first row.
    cnt0: int = 0
    while piece.pixelAt(0, cnt0) == 0:
        cnt0 += 1

    # Subtract from current position, from the column cnt0.
    if col >= cnt0:
        col -= cnt0
    if row + piece.rows() - 1 >= len(currBoard) or col + piece.columns() - 1 >= len(currBoard[0]):
        return False, row, col

    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            if piece.pixelAt(i - row, j - col) != 0 and \
                    (colourMatters and not similarColours(piece.getColour(), colourMap[i][j], dict)):
                return False, row, col
            if currBoard[i][j] + piece.pixelAt(i - row, j - col) > targetBoard[i][j]:
                return False, row, col
    return True, row, col


def nextPos(currBoard: Board, row: int, col: int):
    for i in range(len(currBoard)):
        for j in range(len(currBoard[0])):
            if currBoard[i][j] == 0:
                return i, j
    return -1, -1


def emptyBoard(rows: int, cols: int):
    return [[0 for _ in range(cols)] for _ in range(rows)]


def similarColours(colour1, colour2, dict):
    colour1 = tuple(colour1)
    colour2 = tuple(colour2)
    pair = (colour1, colour2)
    pairRev = (colour2, colour1)
    if pair in dict:
        return dict[pair]

    lab1 = convert_color(sRGBColor(colour1[0], colour1[1], colour1[2]), LabColor)
    lab2 = convert_color(sRGBColor(colour2[0], colour2[1], colour2[2]), LabColor)

    distance = delta_e_cie2000(lab1, lab2)
    ans = distance < COLOURTHRESHOLD
    dict[pair] = ans
    dict[pairRev] = ans
    return ans


def similarColoursJigsaw(colour1, colour2, dict, ct):

    colour1 = tuple(colour1)
    colour2 = tuple(colour2)
    pair = (colour1, colour2)
    pairRev = (colour2, colour1)
    if pair in dict:
        return dict[pair] < ct

    lab1 = convert_color(sRGBColor(colour1[0], colour1[1], colour1[2]), LabColor)
    lab2 = convert_color(sRGBColor(colour2[0], colour2[1], colour2[2]), LabColor)

    distance = delta_e_cie2000(lab1, lab2)
    ans = distance < ct
    dict[pair] = distance
    dict[pairRev] = distance
    return ans


def fillId(matrix, startX, startY, targetId, visitedPoints):
    stack = [(startX, startY)]
    connectedPoints = []

    while stack:
        cX, cY = stack.pop()

        if (cX, cY) in visitedPoints:
            continue
        if not (0 <= cX < matrix.shape[1] and 0 <= cY < matrix.shape[0]):
            continue
        if matrix[cY, cX] != targetId:
            continue

        visitedPoints.add((cX, cY))
        connectedPoints.append((cX, cY))

        neighbors = [(cX + dx, cY + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        stack.extend(neighbors)

    return connectedPoints


def drawPieceOutline(draw, colourMap, pieceId, x, y, cellSize):
    colour = colourMap[pieceId].getColour()
    # Transform to tuples of ints. (The colours were reported as (x.0, y.0, z.0)).
    if isinstance(colour, (tuple, list)):
        colour = tuple(int(c) for c in colour)
    elif isinstance(colour, float):
        colour = int(colour)
    r = [x * cellSize, y * cellSize, (x + 1) * cellSize, (y + 1) * cellSize]
    # Do not use outline as that will come later to shape the pieces.
    draw.rectangle(r, fill=colour, outline=None)

def drawPieceBorder(draw, points, cellSize):
    if not points:
        return

    # Create a mask to draw the border.
    minX = min(p[0] for p in points)
    maxX = max(p[0] for p in points)
    minY = min(p[1] for p in points)
    maxY = max(p[1] for p in points)

    mask = Image.new('L', ((maxX - minX + 1) * cellSize, (maxY - minY + 1) * cellSize), 0)
    maskDraw = ImageDraw.Draw(mask)

    for x, y in points:
        rect = [(x - minX) * cellSize, (y - minY) * cellSize, (x - minX + 1) * cellSize, (y - minY + 1) * cellSize]
        maskDraw.rectangle(rect, fill=1)

    # Convert the mask to a NumPy array and ensure it is of int type.
    maskNp = np.array(mask, dtype=np.uint8) * 255

    # Find the contour of the filled region to later draw the outline.
    contours, _ = cv2.findContours(maskNp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours with black based on cellSize.
    for contour in contours:
        scaledContour = [(point[0][0] + minX * cellSize, point[0][1] + minY * cellSize) for point in contour]
        draw.line(scaledContour + [scaledContour[0]], fill='black', width=2)

# Based on the colourMap and the solution, print the tessellation solution and if needed store it in "outputPath".
def printTessellation(solution, colourMap, cellSize=20, outputPath=None):
    solution = np.array(solution)

    rows, cols = solution.shape
    imgWidth = cols * cellSize
    imgHeight = rows * cellSize

    # Create a blank image with white background.
    image = Image.new('RGB', (imgWidth, imgHeight), 'white')
    draw = ImageDraw.Draw(image)

    # Fill the image with colors.
    for y in range(rows):
        for x in range(cols):
            pieceId = solution[y, x]
            drawPieceOutline(draw, colourMap, pieceId, x, y, cellSize)

    # Draw outlines for each piece using fill algorithm.
    visited = set()
    uniqueIds = np.unique(solution)
    for pieceId in uniqueIds:
        for y in range(rows):
            for x in range(cols):
                if solution[y, x] == pieceId and (x, y) not in visited:
                    points = fillId(solution, x, y, pieceId, visited)
                    drawPieceBorder(draw, points, cellSize)

    if outputPath:
        image.save(outputPath)

    image.show()
    return image