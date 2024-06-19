# Tessellation-Puzzle-Solver
Individual Project - Imperial College London

## Running the program 

To run the program, cd into pythonRecognitionAndGeneration and run: 
python3 .\tps.py --image .\images\puzzle.png

To this add the following flags: \
--show: to have a visual image of the solution. \
--cpp: to opt for the cpp implementation, this works if -- DLX is enabled.\
--DLX n: DLX solution for the solver, n 1 is unoptimised (slow, would not recommend), n = 2 (optimised, works best) \
--BKT n: BKT solution for the solver, n = 1 is for the optimised version. 

Not specifying --cpp to the --DLX flag leads to the python implementation of the DLX algorithm. 

Examples of puzzles that can be solved by TPS are inside the "generated_puzzles" and "images" directories.

To enable matching colours to the target image, add the --colour flag 
## Set up

Make sure all dependencies specified in the requirements.txt file are installed.
Before running the program, head into the "build" directory and run the make command if you want to 
use the C++ version of the DLX algorithm. 

# Mentions

The BKT algorithm is better explained in the supporting report. It consists of depth-first searching
based on the current pieces and state of our puzzle board. 
The DLX python algorithm is us reformatting our problem into a compatible input for the Python dlx library.
The C++ implementation follows the directions of Knuth presented in the "Dancing Links" paper and the solution was adapted 
to our type of problems while taking inspiration from existing sources mentioned in the report. 

## Evaluating

First generate the puzzles running "generator.py". Then, run evaluationScript.py after picking what directories to evaluate
inside script-BKT-Timer, script-DLX-Timer, script-DLX-CPP-Timer. This will give the statistics necessary to recreate the graphs.
Evaluating the jigsaw puzzles is done manually, but statistics are recorded.