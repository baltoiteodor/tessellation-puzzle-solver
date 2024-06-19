import os
import random
from PIL import Image, ImageDraw

# Colour list.
colors = [
    (0, 0, 255),     # Blue
    (255, 0, 0),     # Red
    (255, 255, 0),   # Yellow
    (0, 128, 0),     # Green
    (128, 0, 128),   # Purple
    (139, 69, 19)    # Dark Brown
]

# Helper function to rotate a piece 90 degrees clockwise.
def rotatePiece(piece):
    return [(py, -px) for px, py in piece]

# Generate all unique rotations for a piece.
def generateRotations(piece):
    rotations = []
    current = piece
    for _ in range(4):
        current = rotatePiece(current)
        if current not in rotations:
            rotations.append(current)
    return rotations

# Predefined set of piece types (coordinates relative to (0, 0)).
# Some pieces are written multiple times to change chances of picking pieces.
basePieceTypes = [
    [(0, 0)],  # 1-cell piece
    [(0, 0), (1, 0)],  # 2-cell piece
    [(0, 0), (1, 0), (0, 1)],  # L-shaped piece
    [(0, 0), (1, 0), (0, 1)],  # L-shaped piece
    [(0, 0), (1, 0), (1, 1)],  # 3-cell corner piece
    [(0, 0), (1, 0), (1, 1)],  # 3-cell corner piece
    [(0, 0), (1, 0), (1, 1)],  # 3-cell corner piece
    [(0, 0), (1, 0), (1, 1)],  # 3-cell corner piece
    [(0, 0), (0, 1), (0, 2)],  # 3-cell line piece (horizontal)
    [(0, 0), (1, 0), (2, 0)],  # 3-cell line piece (vertical)
    [(0, 0), (1, 0), (2, 0), (1, 1)],  # T-shaped piece
    [(0, 0), (1, 0), (2, 0), (1, 1)],  # T-shaped piece
    [(0, 0), (1, 0), (2, 0), (1, 1)],  # T-shaped piece
    [(0, 0), (1, 0), (2, 0), (1, 1)],  # T-shaped piece
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],  # Plus-shaped piece
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z-shaped piece
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z-shaped piece
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z-shaped piece
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z-shaped piece
    [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)],  # house-shaped piece
    [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)],   # H-shape piece
    [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)],   # H-shape piece
    [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)],   # H-shape piece
]

# Generate all unique rotations for all piece types.
pieceTypes = []
for basePiece in basePieceTypes:
    pieceTypes.extend(generateRotations(basePiece))

# Separate 1-cell pieces from the rest. These will be used only to fill the board after attempting to place
# a fixed number of random pieces.
singleCellPieces = [piece for piece in pieceTypes if len(piece) == 1]
multiCellPieces = [piece for piece in pieceTypes if len(piece) > 1]

def isValidPiece(piece, grid, x, y, gridSize):
    for px, py in piece:
        nx, ny = x + px, y + py
        if nx < 0 or ny < 0 or nx >= gridSize or ny >= gridSize or grid[nx][ny] != -1:
            return False
    return True

def placePiece(grid, piece, x, y, pieceId):
    for px, py in piece:
        nx, ny = x + px, y + py
        grid[nx][ny] = pieceId

def generatePuzzle(n):
    # Create a larger grid where we put both the board and the pieces.
    gridSize = n * 4
    grid = [[-1 for _ in range(gridSize)] for _ in range(gridSize)]
    pieces = []
    pieceId = 0

    # Place pieces on the board first. Board will be the first nxn square of the grid.
    for i in range(n):
        for j in range(n):
            placedAny = False
            for _ in range(150):  # Attempt to place 100 pieces
                piece = random.choice(multiCellPieces)
                if isValidPiece(piece, grid, i, j, n):
                    placePiece(grid, piece, i, j, pieceId)
                    pieces.append((piece, random.choice(colors)))
                    pieceId += 1
                    placedAny = True
                    break
            # If attempting unsuccessful for 150 times, place 1x1 piece.
            if not placedAny:
                if isValidPiece([(0, 0)], grid, i, j, n):
                    placePiece(grid, [(0, 0)], i, j, pieceId)
                    pieces.append(([(0, 0)], random.choice(colors)))
                    pieceId += 1

    # Ensure all cells in the puzzle grid are filled by filling with 1x1 pieces.
    for i in range(n):
        for j in range(n):
            if grid[i][j] == -1:
                piece = [(0, 0)]
                placePiece(grid, piece, i, j, pieceId)
                pieces.append((piece, random.choice(colors)))
                pieceId += 1

    # Board is done, place the pieces now into the larger grid.
    # Place pieces under the puzzle with at least one cell spacing. Start right below the puzzle.
    startY = n + 1
    piecePositions = set()
    for idx, (piece, color) in enumerate(pieces):

        # Calculate piece dimensions.
        pieceWidth = max(px for px, py in piece) + 1
        pieceHeight = max(py for px, py in piece) + 1

        placed = False
        # Attempt to place the piece in the large grid. Try to find first available position.
        for i in range(startY, gridSize - pieceHeight + 1):
            for j in range(0, gridSize - pieceWidth + 1):
                # Check for all positions to be available first, then check for surroundings, for the pieces to be at
                # least 1 cell from other pieces.
                if all(
                        0 <= i + py < gridSize and 0 <= j + px < gridSize and grid[i + py][j + px] == -1
                        for px, py in piece
                ) and all(
                    0 <= i + py + dy < gridSize and 0 <= j + px + dx < gridSize and grid[i + py + dy][j + px + dx] == -1
                    for dx in range(-1, 2)
                    for dy in range(-1, 2)
                    for px, py in piece
                    if not (dx == 0 and dy == 0)
                ):
                    for px, py in piece:
                        grid[i + py][j + px] = idx  # Fill grid with piece ID.
                    piecePositions.add((i, j))
                    placed = True
                    break
            if placed:
                break

    return grid, pieces

def drawPuzzle(grid, pieces, filename):
    # This determines size of the final image.
    cellSize = 20
    gridSize = len(grid)

    # Calculate sizes.
    imgWidth = gridSize * cellSize
    imgHeight = gridSize * cellSize

    img = Image.new("RGB", (imgWidth, imgHeight), "white")
    draw = ImageDraw.Draw(img)

    # Draw grid without outlines.
    for i in range(gridSize):
        for j in range(gridSize):
            if grid[i][j] != -1:
                # Draw the cell into the image.
                pieceId = grid[i][j]
                color = pieces[pieceId][1]
                x0, y0 = j * cellSize, i * cellSize
                x1, y1 = x0 + cellSize, y0 + cellSize
                draw.rectangle([x0, y0, x1, y1], fill=color)

    img.save(filename)


def generateAndSavePuzzles():
    baseDir = 'generated_puzzles'
    os.makedirs(baseDir, exist_ok=True)
    # Generate 20 puzzles for all sizes from 2 to 19.
    for gridSize in range(2, 20, 1):  # From 2x2 to 19x19.
        gridDir = os.path.join(baseDir, f'{gridSize}x{gridSize}')
        os.makedirs(gridDir, exist_ok=True)
        for i in range(20):
            grid, pieces = generatePuzzle(gridSize)
            filename = os.path.join(gridDir, f'puzzle_{gridSize}x{gridSize}_pieces{len(pieces)}_{i+1}.png')
            drawPuzzle(grid, pieces, filename)

generateAndSavePuzzles()

# Generate and draw one puzzle
# n = 4  # Size of the puzzle grid
# grid, pieces = generatePuzzle(n)
# # print(len(pieces))
# drawPuzzle(grid, pieces, "puzzle_image.png")
