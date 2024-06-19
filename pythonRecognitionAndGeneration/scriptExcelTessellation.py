import os
import pandas as pd

def processFilesToExcel(folderPath, outputExcel, typeValue):
    data = []

    # Iterate over all files in the specified folder.
    for filename in os.listdir(folderPath):
        if filename.endswith(".txt"):
            filePath = os.path.join(folderPath, filename)
            with open(filePath, 'r') as file:
                lines = file.readlines()

                # Extract size n from the filename based on filename format: puzzle_nxn_piecesx_id
                parts = filename.split('_')
                sizePart = parts[1]
                size = sizePart.split('x')[0]

                # Create a dictionary to store the data.
                fileData = {"Size": size, "Type": typeValue}

                # Initialise processing time.
                processingTotalTime = 0.0

                for line in lines[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        label, value = parts
                        if label not in ["Jigsaw Success", "Choosing the best solution for Jigsaw"]:
                            try:
                                value = round(float(value), 5)
                            except ValueError:
                                pass
                            fileData[label] = value

                            # Sum the processing columns into processingTotalTime.
                            if label in ["PreProcessing", "Contour Finding", "Rotating Pieces", "Processing Pieces into Grids"]:
                                processingTotalTime += value

                # Add the aggregate column.
                fileData["Processing Total Time"] = round(processingTotalTime, 5)

                data.append(fileData)

    # Create a DataFrame.
    df = pd.DataFrame(data)

    # Save the data to an Excel file.
    df.to_excel(outputExcel, index=False)

# Specify the folder and output filename.
folderPath = "Evaluation/DLX-CPP/No-Colour"
outputExcel = "TessellationDLXCPPNoColour.xlsx"
typeValue = "DLXCPP_No_Colour"

# Process the files and create the Excel file.
processFilesToExcel(folderPath, outputExcel, typeValue)
