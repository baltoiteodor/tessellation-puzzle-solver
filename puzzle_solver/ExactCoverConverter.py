# This class converts a list of pieces and a board to a matrix representing an exact cover
# problem.
import numpy as np

from misc.piece import Piece
from puzzle_solver.helper import similarColours


class ExactCoverConverter:
    def __init__(self, board: Piece, pieces, colourMap, version, colour):
        # Board will be the biggest piece filled with 1s. We shape the matrix w.r.t it.
        self._pypyColumns = None
        self._pypyRows = None
        self._boardPiece = board
        self._pieces = pieces
        self._matrix = None
        self._rows = 0
        self._cols = 0
        # If version is 0, use home-made DLX, otherwise use the pypy one.
        self._version = version
        self._colouring = colour
        self._colourMap = colourMap

    # def constructMatrix(self):
    #     if self._version == 0:
    #         self._constructMatrixHome()
    #     else:
    #         self._constructMatrixPyPy()

    def constructMatrix(self):
        # The width of the DLX matrix will be the size of the board piece for information on
        # what positions are taken, to which we add the number of pieces.
        boardSize = self._boardPiece.columns() * self._boardPiece.rows()
        piecesNum = len(self._pieces)
        width = boardSize + piecesNum
        self._matrix = []
        self._cols = width
        # Each piece will contribute a number of rows equal to the number of possible
        # rotations it can take * all valid locations in can be placed.

        # This is for PyPy dlx:
        self._pypyColumns = []
        self._pypyRows = []
        if self._version:
            for i in range(self._boardPiece.rows()):
                for j in range(self._boardPiece.columns()):
                    self._pypyColumns.append("r"+str(i) + "c" + str(j))

            for i in range(piecesNum):
                self._pypyColumns.append("P" + str(i + 1))

        for piece in self._pieces:
            numRotations = piece.getRotations()
            for _ in range(numRotations):
                # Current rotation of piece.
                for row in range(self._boardPiece.rows() - piece.rows() + 1):
                    for col in range(self._boardPiece.columns() - piece.columns() + 1):
                        matrixRow = np.zeros(width, dtype=bool)
                        pypyRow = []
                        if self.checkPlace(self._boardPiece.columns(), row, col, piece, matrixRow, pypyRow):
                            # Add a 1 in the matrixRow at the piece orderNum location.
                            matrixRow[boardSize - 1 + piece.orderNum()] = 1
                            if self._version:
                                pypyRow.append(boardSize - 1 + piece.orderNum())
                                self._pypyRows.append(pypyRow)
                            # Add it to the matrix.
                            self._matrix.append(matrixRow)
                            self._rows += 1

                # Rotate piece.
                piece.rotatePiece()

    def checkPlace(self, boardColumns: int, row: int, col: int, piece, matrixRow, pypyRow):
        for r in range(row, row + piece.rows()):
            for c in range(col, col + piece.columns()):
                if self._colouring and (piece.pixelAt(r - row, c - col) != 0 and
                                        not similarColours(piece.getColour(), self._colourMap[r][c])):
                    return False
                if piece.pixelAt(r - row, c - col):
                    # Means colour must be similar.
                    matrixRow[r * boardColumns + c] = 1
                    if self._version:
                        pypyRow.append(r * boardColumns + c)
        return True

    def printMatrix(self):
        if self._version:
            print("Columns for the shit: ", self._pypyColumns)
            print("These rows be for pypy bro: ", self._pypyRows)
            return

        print("Matrix before constructing dancing links.")
        for row in range(self._rows):
            print([int(elem) for elem in self._matrix[row]])

    def getPyPy(self):
        return self._pypyColumns, self._pypyRows

    # def _constructMatrixPyPy(self):
    #     boardSize = self._boardPiece.columns() * self._boardPiece.rows()
    #     width = boardSize + len(self._pieces)
