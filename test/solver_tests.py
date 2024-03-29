import unittest

from misc.piece import Piece
from puzzle_solver.helper import findBoard


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.board = Piece([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        self.pieces = [Piece([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
                       Piece([[1, 1], [1, 1]]),
                       Piece([[1, 1], [1, 1]]),
                       Piece([[1, 1], [1, 1]]),
                       Piece([[1, 1], [1, 1]]),
                       Piece([[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0],
                              [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                              [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0]])]

    def testFindBoard(self):
        self.assertEqual(self.board, findBoard(self.pieces))
