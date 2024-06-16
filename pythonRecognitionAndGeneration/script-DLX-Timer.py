import os
import subprocess
from timeit import default_timer as timer

TIMEOUT_THRESHOLD = 150

folder_paths_colour = [
    # "generated_puzzles/2x2",
    # "generated_puzzles/4x4",
    # "generated_puzzles/6x6",
    # "generated_puzzles/8x8",
    # "generated_puzzles/10x10",
    # "generated_puzzles/3x3",
    "generated_puzzles/5x5",
    "generated_puzzles/7x7",
    "generated_puzzles/9x9",
    # "generated_puzzles/11x11",
    # "generated_puzzles/13x13",
]

folder_paths_no_colour = [
    # "generated_puzzles/2x2",
    # "generated_puzzles/4x4",
    # "generated_puzzles/6x6",
    # "generated_puzzles/8x8",
    # "generated_puzzles/10x10",
    # "generated_puzzles/12x12",
    # "generated_puzzles/14x14",
    # "generated_puzzles/16x16",
    # "generated_puzzles/3x3",
    "generated_puzzles/5x5",
    # "images/test-5",
    # "images/test-6",
    # "images/test-10",
    # "images/test-14",
    # "images/test-15",
    # "images/test-20",
    # "images/test-23",
    # "images/test-30",
    # "images/test-42",
    # "images/test-50",
    # "images/test-65",
]


folder_paths = [
    # "images/test-5",
    # "images/test-6",
    # "images/test-10",
    # "images/test-14",
    # "images/test-15",
    # "images/test-20",
    # "images/test-23",
    # "images/test-30",
    # "images/test-42",
    # "images/test-50",
    # "images/test-65",
    # "images/test-5-nc",
    # "images/test-6-nc",
    # "images/test-10-nc",
    # "images/test-14-nc",
    # "images/test-15-nc",
    # "images/test-20-nc",
    # "images/test-23-nc",
    # "images/test-30-nc",
    # "images/test-42-nc",
    # "images/test-50-nc",
    # "images/test-65-nc",
]

results = {}

def run_command_with_timeout(command, timeout):
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

for folder_path in folder_paths_no_colour:
    # List all files in the folder
    example_files = os.listdir(folder_path)
    totalTime = 0
    numOfRuns = 1
    for run in range(numOfRuns):
        # Iterate over the example files
        for example_file in example_files:
            print(example_file)
            # Construct the command with the current example file as the value for --path flag
            command = [
                'python',
                '.\\tps.py',
                '--image', os.path.join(folder_path, example_file),  # Construct the path for the current example file
                # '--colour',
                '--DLX', '2',
            ]

            print("Running command:", " ".join(command))
            time_taken = run_command_with_timeout(command, TIMEOUT_THRESHOLD)
            if time_taken is not None:
                totalTime += time_taken

    if example_files:
        results[folder_path] = (totalTime / numOfRuns) / len(example_files)
    else:
        results[folder_path] = 0

    print(f"Running {folder_path} on DLX method took on average: ", results[folder_path])


for folder_path in folder_paths_colour:
    # List all files in the folder
    example_files = os.listdir(folder_path)
    totalTime = 0
    numOfRuns = 1
    for run in range(numOfRuns):
        # Iterate over the example files
        for example_file in example_files:
            print(example_file)
            # Construct the command with the current example file as the value for --path flag
            command = [
                'python',
                '.\\tps.py',
                '--image', os.path.join(folder_path, example_file),  # Construct the path for the current example file
                '--colour',
                '--DLX', '2',
            ]

            print("Running command:", " ".join(command))
            time_taken = run_command_with_timeout(command, TIMEOUT_THRESHOLD)
            if time_taken is not None:
                totalTime += time_taken

    if example_files:
        results[folder_path] = (totalTime / numOfRuns) / len(example_files)
    else:
        results[folder_path] = 0

    print(f"Running {folder_path} on DLX method took on average: ", results[folder_path])