# Documentation 

## How does the program work? 

### Main Algorithm 

The aim is to solve puzzles like https://www.doc.ic.ac.uk/~mjw03/PersonalWebpage/Pics/puzzle.jpg, where each puzzle piece can be represented in the form of a multitude of *unit* squares. 

The first challenge of the project is to recognize the shapes from within the image. After correctly finding the shapes and their forms we can continue to the next step. (More on how we recognize shapes in section **Shape recognition**)

Each piece will be first seated straight, with angle 0 relative to the Ox axis (More on this in section **Seating a piece straight**), e.g. <br>

1. Correct: **TODO: Insert correct example.**

2. Incorrect: **TODO: Insert incorrect example.**

After this is complete, each piece will be represented by a matrix that will depict the form of the piece. Entry (i, j) will be 1 in the case when if the grid represented by the matrix is put on top of the piece, there is a part of it inside. Otherwise the value will be 0 (More on this in section **Transforming pieces into matrices**). E.g.: <br>

**TODO: Insert example**

The biggest rectangular shape will be in our case the box. The matrix corresponding to it will be filled with 0s and our algotithm's aim is to fill this with 1s only using the pieces found in the image. 

## Potential Extensions: 

1. 3D puzzles
2. Round shapes / triangles 