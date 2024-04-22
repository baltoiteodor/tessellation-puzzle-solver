
# Class for each 1 element in the matrix. This object will contain information about
# other 1s to it's left, right, up and down, as well as the list header.
class MatrixElement:

    def __init__(self, idElem: int):
        self._L = idElem
        self._R = idElem
        self._U = idElem
        self._D = idElem
        self._C = idElem
        self._id = idElem

    def setLeft(self, other):
        # other in this case will always be a MatrixElement id.
        self._L = other

    def setRight(self, other):
        # other in this case will always be a MatrixElement id.
        self._R = other

    def setUp(self, other):
        # other in this case will always be a MatrixElement id.
        self._U = other

    def setDown(self, other):
        # other in this case will always be a MatrixElement id.
        self._D = other

    def setListHeader(self, other):
        # other in this case will always be a MatrixElement id.
        self._C = other

    def left(self):
        return self._L

    def right(self):
        return self._R

    def up(self):
        return self._U

    def down(self):
        return self._D

    def listHeader(self):
        return self._C

