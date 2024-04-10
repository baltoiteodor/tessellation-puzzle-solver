import os
import subprocess

# Path to the folder containing examples
folder_path = "images\\test"

# List all files in the folder
example_files = os.listdir(folder_path)

# Iterate over the example files
for example_file in example_files:
    print(example_file)
    # Construct the command with the current example file as the value for --path flag
    command = [
        'python',
        '.\\tps.py',
        '--image', os.path.join(folder_path, example_file),  # Construct the path for the current example file
        '--colour',
        '--DLX', '1',
    ]

    print("Running command:", " ".join(command))  # Print the command being executed

    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
