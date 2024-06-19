import os
import pandas as pd

# This script creates an excel from a list of jigsaw results.
def processFilesToExcel(folderPath, outputExcel):
    data = []

    # Iterate over all files in the specified folder.
    for filename in os.listdir(folderPath):
        if filename.endswith(".txt"):
            filePath = os.path.join(folderPath, filename)
            with open(filePath, 'r') as file:
                lines = file.readlines()

                # Extract the number of pieces from the filename (last two digits before .txt).
                # Naming convention: puzzle54.txt
                pieces = filename[-6:-4]

                # Create a dictionary to store the data for this file.
                fileData = {"Name": filename[:-4], "Pieces": pieces}  # Remove .txt extension for the name.

                for line in lines[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        label, value = parts
                        fileData[label] = value

                data.append(fileData)

    # Create a DataFrame from the collected data.
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file.
    df.to_excel(outputExcel, index=False)

# Specify the folder from where to gather the data and where to save the excel.
folderPath = "Evaluation/DLX-CPP/Jigsaw"
outputExcel = "JigsawResults.xlsx"

# Process the files and create the Excel file
processFilesToExcel(folderPath, outputExcel)
