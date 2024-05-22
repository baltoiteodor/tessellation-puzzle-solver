import os
import subprocess
from timeit import default_timer as timer


# Path to the folder containing examples
folder_path = "images/test-42"

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
            '--cpp'
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

print("Running low_num_tests on DLX-CPP method took on average: ", totalTime / numOfRuns)