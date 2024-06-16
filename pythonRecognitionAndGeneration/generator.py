import os
import random
from PIL import Image, ImageDraw

# Define colors
colors = [
    (0, 0, 255),     # Blue
    (255, 0, 0),     # Red
    (255, 255, 0),   # Yellow
    (0, 128, 0),     # Green
    (128, 0, 128),   # Purple
    (139, 69, 19)    # Dark Brown
]

# Helper function to rotate a piece 90 degrees clockwise
def rotatePiece(piece):
    return [(py, -px) for px, py in piece]

# Generate all unique rotations for a piece
def generateRotations(piece):
    rotations = []
    current = piece
    for _ in range(4):
        current = rotatePiece(current)
        if current not in rotations:
            rotations.append(current)
    return rotations

# Predefined set of piece types (coordinates relative to (0, 0))
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

# Generate all unique rotations for all piece types
pieceTypes = []
for basePiece in basePieceTypes:
    pieceTypes.extend(generateRotations(basePiece))

# Separate 1-cell pieces from the rest
singleCellPieces = [piece for piece in pieceTypes if len(piece) == 1]
multiCellPieces = [piece for piece in pieceTypes if len(piece) > 1]

def isValidPiece(piece, grid, x, y, grid_size):
    for px, py in piece:
        nx, ny = x + px, y + py
        if nx < 0 or ny < 0 or nx >= grid_size or ny >= grid_size or grid[nx][ny] != -1:
            return False
    return True

def placePiece(grid, piece, x, y, pieceId):
    for px, py in piece:
        nx, ny = x + px, y + py
        grid[nx][ny] = pieceId

def generatePuzzle(n):
    grid_size = n * 4  # Create a larger grid with reduced size to avoid excessive height
    grid = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
    pieces = []
    pieceId = 0

    # Place pieces in the puzzle area
    for i in range(n):
        for j in range(n):
            placed_any = False
            for _ in range(150):  # Attempt to place 100 pieces
                piece = random.choice(multiCellPieces)
                if isValidPiece(piece, grid, i, j, n):
                    placePiece(grid, piece, i, j, pieceId)
                    pieces.append((piece, random.choice(colors)))
                    pieceId += 1
                    placed_any = True
                    break
            if not placed_any:
                if isValidPiece([(0, 0)], grid, i, j, n):
                    placePiece(grid, [(0, 0)], i, j, pieceId)
                    pieces.append(([(0, 0)], random.choice(colors)))
                    pieceId += 1

    # Ensure all cells in the puzzle grid are filled
    for i in range(n):
        for j in range(n):
            if grid[i][j] == -1:
                piece = [(0, 0)]
                placePiece(grid, piece, i, j, pieceId)
                pieces.append((piece, random.choice(colors)))
                pieceId += 1

    # Place pieces under the puzzle with at least one cell spacing
    startY = n + 1  # Start placing pieces below the puzzle
    piece_positions = set()
    for idx, (piece, color) in enumerate(pieces):
        piece_width = max(px for px, py in piece) + 1
        piece_height = max(py for px, py in piece) + 1
        # print("Trying this: ", piece)

        placed = False
        for i in range(startY, grid_size - piece_height + 1):
            for j in range(0, grid_size - piece_width + 1):
                if all(
                        0 <= i + py < grid_size and 0 <= j + px < grid_size and grid[i + py][j + px] == -1
                        for px, py in piece
                ) and all(
                    0 <= i + py + dy < grid_size and 0 <= j + px + dx < grid_size and grid[i + py + dy][j + px + dx] == -1
                    for dx in range(-1, 2)
                    for dy in range(-1, 2)
                    for px, py in piece
                    if not (dx == 0 and dy == 0)
                ):
                    for px, py in piece:
                        grid[i + py][j + px] = idx  # Use idx to correctly reference pieces
                    piece_positions.add((i, j))
                    placed = True
                    # print("Hurray", piece)
                    break
            if placed:
                break

    return grid, pieces

def drawPuzzle(grid, pieces, filename):
    cellSize = 20  # Reduced cell size to make the image smaller
    grid_size = len(grid)
    imgWidth = grid_size * cellSize
    # for row in grid:
    #     print(row)
    # Calculate required height
    total_piece_height = sum(max(py for px, py in piece) + 1 for piece, color in pieces)
    # imgHeight = (n + total_piece_height) * cellSize
    imgHeight = grid_size * cellSize

    img = Image.new("RGB", (imgWidth, imgHeight), "white")
    draw = ImageDraw.Draw(img)

    # Draw grid without outlines
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] != -1:
                pieceId = grid[i][j]
                color = pieces[pieceId][1]
                x0, y0 = j * cellSize, i * cellSize
                x1, y1 = x0 + cellSize, y0 + cellSize
                draw.rectangle([x0, y0, x1, y1], fill=color)

    img.save(filename)
    # img.show()


def generateAndSavePuzzles():
    base_dir = 'generated_puzzles'
    os.makedirs(base_dir, exist_ok=True)
    for grid_size in range(11, 20, 2):  # From 11x11 to 19x19
        grid_dir = os.path.join(base_dir, f'{grid_size}x{grid_size}')
        os.makedirs(grid_dir, exist_ok=True)
        for i in range(20):
            grid, pieces = generatePuzzle(grid_size)
            filename = os.path.join(grid_dir, f'puzzle_{grid_size}x{grid_size}_pieces{len(pieces)}_{i+1}.png')
            drawPuzzle(grid, pieces, filename)

generateAndSavePuzzles()

# Generate and draw one puzzle
# n = 4  # Size of the puzzle grid
# grid, pieces = generatePuzzle(n)
# # print(len(pieces))
# drawPuzzle(grid, pieces, "puzzle_image.png")
