import os
import subprocess

folderPath = "images/test_low_num"
exampleFiles = os.listdir(folderPath)

for exampleFile in exampleFiles:
    print(exampleFile)

    # Construct the command with the current example file as the value for --path flag.
    command = [
        'python',
        os.path.join('.', 'tps.py'),
        '--image', os.path.join(folderPath, exampleFile),
        '--colour',
        '--DLX', '1',
    ]

    print("Running command:", " ".join(command))

    # Run.
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
