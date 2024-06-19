import os
import subprocess
from timeit import default_timer as timer

TIMEOUTTHRESHOLD = 150

# Uncomment what folders to solve from.
folderPathsColour = [
    # "generated_puzzles/2x2",
    # "generated_puzzles/4x4",
    # "generated_puzzles/6x6",
    # "generated_puzzles/8x8",
    # "generated_puzzles/10x10",
    # "generated_puzzles/12x12",
    # "generated_puzzles/14x14",
    # "generated_puzzles/16x16",
    # "generated_puzzles/18x18",
    "generated_puzzles/20x20",
    # "generated_puzzles/21x21",
    # "generated_puzzles/22x22",
    # "generated_puzzles/24x24",
    # "generated_puzzles/3x3",
    # "generated_puzzles/5x5",
    # "generated_puzzles/7x7",
    # "generated_puzzles/9x9",
    # "generated_puzzles/11x11",
    # "generated_puzzles/13x13",
    # "generated_puzzles/15x15",
    # "generated_puzzles/17x17",
    # "generated_puzzles/19x19",
]

folderPathsNoColour = [
    # "generated_puzzles/2x2",
    # "generated_puzzles/4x4",
    # "generated_puzzles/6x6",
    # "generated_puzzles/8x8",
    # "generated_puzzles/10x10",
    # "generated_puzzles/12x12",
    # "generated_puzzles/14x14",
    # "generated_puzzles/16x16",
    # "generated_puzzles/18x18",
    # "generated_puzzles/20x20",
    # "generated_puzzles/3x3",
    # "generated_puzzles/5x5",
    # "generated_puzzles/7x7",
    # "generated_puzzles/9x9",
    # "generated_puzzles/11x11",
    # "generated_puzzles/13x13",
]

results = {}

def runCommandWithTimeout(command, timeout):
    try:
        timeStart = timer()
        subprocess.run(command, check=True, timeout=timeout)
        timeStop = timer()
        return timeStop - timeStart
    except subprocess.TimeoutExpired:
        print(f"Command exceeded timeout of {timeout} seconds and was terminated.")
        return timeout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return None


for folderPath in folderPathsNoColour:
    exampleFiles = os.listdir(folderPath)
    totalTime = 0
    numOfRuns = 1
    for run in range(numOfRuns):
        # Iterate over the files in the directory and run the command.
        for exampleFile in exampleFiles:
            print(exampleFile)
            command = [
                'python',
                '.\\tps.py',
                '--image', os.path.join(folderPath, exampleFile),
                # '--colour',
                '--DLX', '2',
                '--cpp'
            ]

            print("Running command:", " ".join(command))
            timeTaken = runCommandWithTimeout(command, TIMEOUTTHRESHOLD)
            if timeTaken is not None:
                totalTime += timeTaken

    if exampleFiles:
        results[folderPath] = (totalTime / numOfRuns) / len(exampleFiles)
    else:
        results[folderPath] = 0

    print(f"Running {folderPath} on DLX-CPP method took on average: ", results[folderPath])

for folderPath in folderPathsColour:
    exampleFiles = os.listdir(folderPath)
    totalTime = 0
    numOfRuns = 1
    for run in range(numOfRuns):
        # Iterate over the files and run the command.
        for exampleFile in exampleFiles:
            print(exampleFile)
            command = [
                'python',
                '.\\tps.py',
                '--image', os.path.join(folderPath, exampleFile),
                '--colour',
                '--DLX', '2',
                '--cpp'
            ]

            print("Running command:", " ".join(command))
            timeTaken = runCommandWithTimeout(command, TIMEOUTTHRESHOLD)
            if timeTaken is not None:
                totalTime += timeTaken

    if exampleFiles:
        results[folderPath] = (totalTime / numOfRuns) / len(exampleFiles)
    else:
        results[folderPath] = 0

    print(f"Running {folderPath} on DLX-CPP method took on average: ", results[folderPath])