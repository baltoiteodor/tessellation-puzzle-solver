import os
import subprocess
from timeit import default_timer as timer

folder_paths = [
    # "images/test-5",
    # "images/test-6",
    # "images/test-10",
    # "images/test-14",
    # "images/test-15",
    "images/test-20",
    # "images/test-23",
    # "images/test-30",
    # "images/test-42",
    # "images/test-50",
    # "images/test-65",
]

results = {}

for folder_path in folder_paths:
    # Path to the folder containing examples
    # folder_path = "images/test-20"

    # List all files in the folder
    example_files = os.listdir(folder_path)
    totalTime = 0
    numOfRuns = 3
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
                '--BKT', '1',
                # '--show'
            ]

            print("Running command:", " ".join(command))  # Print the command being executed
            # Run the command
            try:
                timeStart = timer()
                subprocess.run(command, check=True)
                timeStop = timer()
                totalTime += timeStop - timeStart
            except subprocess.CalledProcessError as e:
                print("Error:", e)

    results[folder_path] = (totalTime / numOfRuns) / len(example_files)
    print(f"Running {folder_path} on BKT method took on average: ", (totalTime / numOfRuns) / len(example_files))

for folder_path in folder_paths:
    print(f"Running {folder_path} on BKT method took on average: ", results[folder_path])
