import cv2
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000

from misc.piece import *


def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)

# COLOURTHRESHOLD = 20

# Try so of the X cells is less than COLOURTHRESHOLD and their sum is less than COLOURSUMTHRESHOLD.
# COLOURSUMTHRESHOLD = 60

def resizeToDimensions(image, contour, target_width, target_height):
    # Convert the contour to a numpy array if it isn't already
    contour_np = np.array(contour, dtype=np.int32)

    # Calculate the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour_np)

    # Calculate separate scale factors for width and height
    width_scale = target_width / w
    height_scale = target_height / h

    # Scale the contour by applying different scale factors to x and y coordinates
    scaled_contour = np.column_stack((
        (contour_np[:, :, 0] - x) * width_scale,
        (contour_np[:, :, 1] - y) * height_scale
    )).astype(np.int32)

    # Extract the region of interest (ROI) from the original image using the bounding box
    roi = image[y:y+h, x:x+w]
    cv2.imwrite('Notscaled.png', roi)

    # Resize the ROI to the target dimensions
    scaled_roi = cv2.resize(roi, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Create a mask for the scaled contour
    scaled_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    cv2.drawContours(scaled_mask, [scaled_contour], -1, (255), thickness=cv2.FILLED)

    # Apply the scaled mask to the scaled ROI
    scaled_roi_with_mask = cv2.bitwise_and(scaled_roi, scaled_roi, mask=scaled_mask)

    # Save or return the scaled ROI with mask as an image
    cv2.imwrite('scaled.png', scaled_roi_with_mask)
    return scaled_contour, scaled_roi_with_mask

def scalePiece(piece: Piece, scaleFactor, image):
    originalContour = piece.getOriginalContour().getContour()
    # print("LL", piece.getOriginalContour().getArea())

    originalContournp = np.array(originalContour, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(originalContournp)

    # Resize the bounding box dimensions
    scaledW = int(w * scaleFactor)
    scaledH = int(h * scaleFactor)

    # Scale and translate the contour
    scaledContour = ((originalContournp - (x, y)) * scaleFactor).astype(np.int32)

    # Create a mask for the contour
    mask = np.zeros_like(image[:,:,0])
    cv2.drawContours(mask, [scaledContour], -1, (255), thickness=cv2.FILLED)

    # Extract the region of interest (ROI) from the original image using the bounding box
    roi = image[y:y+h, x:x+w]

    # Resize the ROI using the scaled dimensions
    scaledRoi = cv2.resize(roi, (scaledW, scaledH), interpolation=cv2.INTER_LINEAR)

    # Create a mask for the scaled contour
    scaledMask = np.zeros_like(scaledRoi[:,:,0])
    cv2.drawContours(scaledMask, [scaledContour], -1, (255), thickness=cv2.FILLED)

    # Apply the scaled mask to the scaled ROI
    scaledRoiWithMask = cv2.bitwise_and(scaledRoi, scaledRoi, mask=scaledMask)

    # Save the scaled ROI with mask as an image
    cv2.imwrite('scaled.png', scaledRoiWithMask)

    # Transform into Contour with new image.
    newContour = Contour(scaledContour, scaledRoiWithMask, piece.orderNum())
    unitLen = piece.getUnitLen()

    # Create new grid. If the error is too great return failure to scale the board.
    coveredArea = 0.0
    pieceArea = newContour.getArea()

    box = cv2.boxPoints(newContour.getMinAreaRect())
    box = np.int0(box)
    # x is width, y is height.
    topLeftX = np.min(box[:, 0])
    topLeftY = np.min(box[:, 1])
    botRightX = np.max(box[:, 0])
    botRightY = np.max(box[:, 1])

    # For each unit of the grid, check if the centre is inside the polygon, if yes, then put 1 inside the
    # grid, otherwise 0.
    # Start with row 0, stop when we are outside the rectangle. Same for columns.
    unitX = topLeftX
    indexX = 0
    # print("Is same width?: ", width, botRightX-topLeftX)
    # Due to width/height switching I will calculate my own.
    width = botRightX - topLeftX
    height = botRightY - topLeftY
    rows = int(width / unitLen + 1)
    cols = int(height / unitLen + 1)
    # Invert the x, y to y, x in the grid, so it looks like in the image.
    grid = np.zeros((cols, rows))
    # colours = [[(0.0, 0.0, 0.0) for _ in range(rows)] for _ in range(cols)]
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
    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour.reshape(-1, 1, 2)
    elif contour.ndim != 3:
        raise ValueError("Contour must be a 3-dimensional array with shape (n, 1, 2).")
    # Initialize minimum distance and the closest point
    min_dist = float('inf')
    closest_point = None

    # Iterate over each point in the contour
    for contour_point in contour:
        # Calculate the Euclidean distance between the given point and the current contour point
        dist = np.linalg.norm(contour_point[0] - point)

        # Update minimum distance and closest point if current distance is smaller
        if dist < min_dist:
            min_dist = dist
            closest_point = tuple(contour_point[0])

    return closest_point

def findClosestPointOnMask(mask, target_point):
    # Get indices where the mask is non-zero (i.e., points belonging to the contour)
    y_indices, x_indices = np.nonzero(mask)
    contour_points = np.column_stack((x_indices, y_indices))

    # Calculate distances from each contour point to the target point
    distances = np.sqrt((contour_points[:, 0] - target_point[0])**2 + (contour_points[:, 1] - target_point[1])**2)

    # Get the index of the closest point
    closest_point_index = np.argmin(distances)
    closest_point = contour_points[closest_point_index]

    return tuple(closest_point)


# 1st argument: 2D array with indexes for the pieces.
# 2nd argument: dictionary from index of piece to the corresponding piece.
# Construct the image solution of the puzzle.

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
def printJigsawBad(outputMatrix, dictToPieces, originalImage):
    # Map pieces to their top left corner, as well as establish a relation between the corner and the top left of the
    # piece bounding rectangle.
    dictToLeftCorners = {}
    dictToRightCorners = {}
    dictToMoveVectorsRect = {}
    # Distance from right corner of jigsaw to left corner of jigsaw.
    dictToMoveVectorsPiece = {}

    for idx in dictToPieces.keys():
        currentPiece = dictToPieces[idx]
        x, y, w, h = currentPiece.getOriginalContour().getBoundingRect()
        topLeftRect = (x, y)
        # TODO: Calculate this after piece retrieval bcuz they will be rotated. Rethink this shit.

        topRightRect = (x + w, y)
        botLeftRect = (x, y + h)
        botRightRect = (x + w, y + h)
        topLeftCorner = findClosestContourPoint(currentPiece.getOriginalContour().getContour(), np.array(topLeftRect))
        topRightCorner = findClosestContourPoint(currentPiece.getOriginalContour().getContour(), np.array(topRightRect))
        moveVectorRect = np.array(topLeftCorner) - np.array(topLeftRect)
        moveVectorPiece = np.array(topRightCorner) - np.array(topLeftCorner)

        # if moveVectorPiece[1] > 0:
        #
        #     img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        #
        #     # Draw the contour
        #     cv2.drawContours(img, [currentPiece.getOriginalContour().getContour() - np.array([[600, 1200]])], -1, (0, 255, 0), 2)  # Green for the contour
        #
        #     # Draw circles for the corners
        #     cv2.circle(img, topLeftCorner - np.array((600, 1200)), 5, (255, 0, 0), -1)  # Red for topLeftCorner
        #     cv2.circle(img, topRightCorner - np.array((600, 1200)), 5, (0, 0, 255), -1)  # Blue for topRightCorner
        #     window_name = f"Contour {idx}"
        #     cv2.imshow(window_name, img)
        #     cv2.waitKey(0)


        print("helo here: ", moveVectorPiece, moveVectorRect)
        # TODO: save this info.
        dictToLeftCorners[idx] = topLeftCorner
        dictToRightCorners[idx] = topRightCorner
        dictToMoveVectorsRect[idx] = moveVectorRect
        dictToMoveVectorsPiece[idx] = moveVectorPiece

    # Next we will use the top left corners of pieces to place the jigsaw pieces in the new, originally black, image, by
    # seeing what piece we need to place next based on the outputMatrix and previous placed pieces that will give the new
    # locations of where top left corners should be placed. We will then use the move vectors to find the bounding rectangle
    # position and use the createROI function to place the piece at that location.

    # Starting to build the image.
    # print("Hmmmm: ", originalImage.shape)
    solutionImage = np.zeros(originalImage.shape, dtype=np.uint8)
    nextTopLeft = np.array((0, 0))
    # Will use this dictionary to mark what pieces were already placed when iterating through the outputMatrix.
    piecesDone = {}
    piecesDone[0] = True
    for row in range(len(outputMatrix)):
        for col in range(len(outputMatrix[row])):
            if not (outputMatrix[row][col] in piecesDone):
                if row > 0:
                    cv2.imwrite("progress.png", solutionImage)
                    return
                pieceId = outputMatrix[row][col]
                # Fill shape of piece in the output matrix.
                pieceShape = retrieveShape(outputMatrix, pieceId, row, col)
                # print("Here is a piece shape extracted: ", pieceShape)
                currentPiece = dictToPieces[pieceId]
                angle = currentPiece.retrieveAngle(pieceShape)
                # print("Angle: ", angle)
                # print("Current piece and stuff: ", pieceId, row, col, nextTopLeft)
                print(dictToPieces[pieceId])
                piecesDone[pieceId] = True
                # corner = dictToCorners[pieceId]
                targetLocation = nextTopLeft - dictToMoveVectorsRect[pieceId]
                print("Target Location: ", targetLocation)
                currContour = currentPiece.getOriginalContour()
                currContour.createROIAtAngle(targetLocation, solutionImage, angle)
                # Calculate the new nextTopLeft somehow. Might be top right of the piece, aka the closest point
                # to the top right of the bounding rectangle. Add move vector from the 2 corners to the nextTopLeft probably?
                nextTopLeft += dictToMoveVectorsPiece[pieceId]
                cv2.imwrite("progress.png", solutionImage)


    cv2.imwrite("progress.png", solutionImage)


def rotateImage(image, angle, center):
    # cv2.imwrite(f"checker{angle}1.png", image)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    rotated_img = cv2.warpAffine(image, matrix, (new_width, new_height))
    # cv2.imwrite(f"checker{angle}2.png", rotated_img)

    return rotated_img

def placeAndRotateContour(contour, angle, original_img, target_img, nextTopLeft):
    # Create a mask and extract the ROI
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    roi = cv2.bitwise_and(original_img, original_img, mask=mask)
    extracted = roi[y:y+h, x:x+w]
    mask_extracted = mask[y:y+h, x:x+w]
    # print("Size mask: ", mask_extracted.shape)
    # cv2.imwrite(f"maPrior{angle}.png", mask_extracted)

    center = (w // 2, h // 2)
    rotated_img = rotateImage(extracted, angle, center)
    rotated_mask = rotateImage(mask_extracted, angle, center)

    # print("Size rot mask: ", rotated_mask.shape)
    non_zero_points = np.column_stack(np.where(rotated_mask > 0))
    min_x, min_y = non_zero_points.min(axis=0)
    max_x, max_y = non_zero_points.max(axis=0)

    topLeftRect = (min_y, min_x)
    topRightRect = (max_y, min_x)
    botLeftRect = (min_y, max_x)
    # cv2.imwrite(f"ma{angle}.png", rotated_mask)
    topLeftCorner = findClosestPointOnMask(rotated_mask, np.array(topLeftRect))
    topRightCorner = findClosestPointOnMask(rotated_mask, np.array(topRightRect))
    botLeftCorner = findClosestPointOnMask(rotated_mask, np.array(botLeftRect))
    # print("Current target: ", nextTopLeft)
    # print("offset", topLeftRect[0] - topLeftCorner[0], topLeftRect[1] - topLeftCorner[1])
    # print("Bounding: ", botLeftRect, topLeftRect, topRightRect)
    # print(topLeftCorner, topRightCorner)

    placement_x = nextTopLeft[0] + (topLeftRect[0] - topLeftCorner[0])
    placement_y = nextTopLeft[1] + (topLeftRect[1] - topLeftCorner[1])

    placement_x = max(placement_x, 0)
    placement_y = max(placement_y, 0)
    # print("Pl: ", placement_x, placement_y)
    # print("added to this: ", rotated_img.shape[1], rotated_img.shape[0])
    # print("Will be inverted in target area.")
    target_area = target_img[placement_y:placement_y + rotated_img.shape[0], placement_x:placement_x + rotated_img.shape[1]]
    # print("targetArea: ", (target_area.shape[1], target_area.shape[0]))
    mask_resized = cv2.resize(rotated_mask, (target_area.shape[1], target_area.shape[0]))

    target_region_masked = cv2.bitwise_and(target_area, target_area, mask=cv2.bitwise_not(mask_resized))
    final_region = cv2.add(target_region_masked, cv2.bitwise_and(rotated_img, rotated_img, mask=mask_resized))

    target_img[placement_y:placement_y + rotated_img.shape[0], placement_x:placement_x + rotated_img.shape[1]] = final_region

    newNextTopLeft = (nextTopLeft[0] + (topRightCorner[0] - topLeftCorner[0]), nextTopLeft[1] + (topRightCorner[1] - topLeftCorner[1]))
    newNextBotLeft = (nextTopLeft[0] + (botLeftCorner[0] - topLeftCorner[0]), nextTopLeft[1] + (botLeftCorner[1] - topLeftCorner[1]))

    return newNextTopLeft, newNextBotLeft

def getNextRow(outputMatrix, row):
    for r in range(row, len(outputMatrix)):
        if outputMatrix[r][0] != 0:
            return r
    return -1
def printJigsaw(outputMatrix, dictToPieces, originalImage, colourMap):

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
                # if row > 0:
                #     cv2.imwrite("progress.png", solutionImage)
                #     return
                pieceId = outputMatrix[row][col]
                # Fill shape of piece in the output matrix.
                # TODO: retrieve the colour map of the shape as well.
                pieceShape, pieceColourGrid = retrieveShape(outputMatrix, pieceId, row, col, colourMap)
                print("Here is a piece shape extracted: ", pieceShape)
                currentPiece = dictToPieces[pieceId]
                if pieceId == 5:
                    plt.imshow(pieceColourGrid)
                    plt.axis('off')  # Turn off axis labels
                    plt.show()
                angle = currentPiece.retrieveAngle(pieceShape, pieceColourGrid)
                print("Angle: ", angle)
                print("Current piece and stuff: ", pieceId, row, col, nextTopLeft)
                piecesDone[pieceId] = True

                # TODO: Might want to do alat in a go, not separate, idt it s possible.
                # Get the piece contour and rotate it at the desired angle.
                currContour = currentPiece.getOriginalContour().getContour()
                if beenThru == 0:
                    # print("HARO EVERYNIAN: ", row, nextTopLeft)
                    nextTopLeft, botLeft = placeAndRotateContour(currContour, 360 - angle, currentPiece.getOriginalContour().getImage(), solutionImage, nextTopLeft)
                else:
                    # print("HARO EVERYNIAN: ", row, botLefts[currIndex])
                    nextTopLeft, botLeft = placeAndRotateContour(currContour, 360 - angle, currentPiece.getOriginalContour().getImage(), solutionImage, botLefts[currIndex])
                # print("mm: ", nextTopLeft)
                # print("mmBotLeft: ", botLeft)
                botLefts[currIndex] = botLeft
                currIndex += 1
        beenThru += 1

    cv2.imwrite("progress.png", solutionImage)

def trimGrid(grid):
    # Remove the last columns if all zero.
    while np.all(grid[:, -1] == 0):
        grid = grid[:, :-1]
    # Remove leading columns with all zeros
    while np.all(grid[:, 0] == 0):
        grid = grid[:, 1:]
    # Remove the last row if all zero.
    while np.all(grid[-1, :] == 0):
        grid = grid[:-1, :]
    # Remove leading rows with all zeros
    while np.all(grid[0, :] == 0):
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


def removePiece(currBoard: Board, piece: Piece, row: int, col: int):
    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            currBoard[i][j] -= piece.pixelAt(i - row, j - col)


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
    # print(rotatedGrid)
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
def isValid(currBoard: Board, targetBoard: Board, colourMap, piece: Piece, row: int, col: int, colourMatters: bool):
    # Pieces will have leading 0s in the matrix like the + sign. In this case, change the row, piece of where to put
    # the piece by the leading amount of 0s on the first row. (I think)
    cnt0: int = 0
    while piece.pixelAt(0, cnt0) == 0:
        cnt0 += 1

    # Subtract from current position, from the column cnt0.
    # print("Before: ", row, col)
    if col >= cnt0:
        col -= cnt0
    # print("After: ", row, col, cnt0)
    if row + piece.rows() - 1 >= len(currBoard) or col + piece.columns() - 1 >= len(currBoard[0]):
        return False, row, col

    for i in range(row, row + piece.rows()):
        for j in range(col, col + piece.columns()):
            if piece.pixelAt(i - row, j - col) != 0 and \
                    (colourMatters and not similarColours(piece.getColour(), colourMap[i][j], {})):
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
    # print(colour1)
    # print(colour2)
    colour1 = tuple(colour1)
    colour2 = tuple(colour2)
    pair = (colour1, colour2)
    pairRev = (colour2, colour1)
    if pair in dict:
        return dict[pair]

    lab1 = convert_color(sRGBColor(colour1[0], colour1[1], colour1[2]), LabColor)
    lab2 = convert_color(sRGBColor(colour2[0], colour2[1], colour2[2]), LabColor)
    # lab1 = LabColor(colour1[0], colour1[1], colour1[2])
    # lab2 = LabColor(colour2[0], colour2[1], colour2[2])
    distance = delta_e_cie2000(lab1, lab2)
    ans = distance < COLOURTHRESHOLD
    dict[pair] = ans
    dict[pairRev] = ans
    return ans

def similarColoursJigsaw(colour1, colour2, dict, COLOURTHRESHOLD):
    # print(colour1)
    # print(colour2)
    colour1 = tuple(colour1)
    colour2 = tuple(colour2)
    pair = (colour1, colour2)
    pairRev = (colour2, colour1)
    if pair in dict:
        return dict[pair] < COLOURTHRESHOLD

    lab1 = convert_color(sRGBColor(colour1[2], colour1[1], colour1[0]), LabColor)
    lab2 = convert_color(sRGBColor(colour2[2], colour2[1], colour2[0]), LabColor)
    # lab1 = LabColor(colour1[0], colour1[1], colour1[2])
    # lab2 = LabColor(colour2[0], colour2[1], colour2[2])
    distance = delta_e_cie2000(lab1, lab2)
    ans = distance < COLOURTHRESHOLD
    dict[pair] = distance
    dict[pairRev] = distance
    return ans
