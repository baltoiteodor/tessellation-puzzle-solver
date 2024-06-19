import os
import pandas as pd
from scipy import stats
import numpy as np

def processFilesToExcel(folderPath, outputExcel, typeValue):
    data = []

    # Iterate over all files in the specified folder.
    for filename in os.listdir(folderPath):
        if filename.endswith(".txt"):
            filePath = os.path.join(folderPath, filename)
            with open(filePath, 'r') as file:
                lines = file.readlines()

                # Extract data from the filename.
                parts = filename.split('_')
                sizePart = parts[1]
                size = sizePart.split('x')[0]

                # Create a dictionary to store data.
                fileData = {"Size": size, "Type": typeValue}

                # This is for aggregate time spent processing.
                processingTotalTime = 0.0

                for line in lines[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        label, value = parts
                        # Ignore these.
                        if label not in ["Jigsaw Success", "Choosing the best solution for Jigsaw"]:
                            try:
                                value = round(float(value), 5)
                            except ValueError:
                                pass
                            fileData[label] = value

                            # Sum the specific columns into processingTotalTime.
                            if label in ["PreProcessing", "Contour Finding", "Rotating Pieces", "Processing Pieces into Grids"]:
                                processingTotalTime += value

                fileData["Processing Total Time"] = round(processingTotalTime, 5)

                # Remove columns that were aggregated.
                for key in ["PreProcessing", "Contour Finding", "Rotating Pieces", "Processing Pieces into Grids"]:
                    if key in fileData:
                        del fileData[key]

                data.append(fileData)

    # Create a DataFrame from the collected data.
    df = pd.DataFrame(data)

    # Calculate statistics.
    grouped = df.groupby('Size')
    statsData = []

    for name, group in grouped:
        row = {"Size": name, "Type": typeValue}

        # Ensure there are at least 20 files in the group. If not fill with Threshold plus 30 = 180.
        if len(group) < 20:
            fillCount = 20 - len(group)
            numericColumns = group.select_dtypes(include=[np.number]).columns
            groupMean = group[numericColumns].mean()
            fillRows = [groupMean] * fillCount
            fillDF = pd.DataFrame(fillRows)
            fillDF["Total time"] = 180
            group = pd.concat([group, fillDF], ignore_index=True)

        if "Total time" in group.columns:
            values = group["Total time"].dropna().values

            # Calculate original mean and percentiles.
            originalMean = round(values.mean(), 6)
            percentile50 = round(np.percentile(values, 50), 6)
            percentile90 = round(np.percentile(values, 90), 6)
            percentile99 = round(np.percentile(values, 99), 6)

            # Remove outliers using Z-score method.
            zScores = stats.zscore(values)
            filteredValues = values[(zScores > -2) & (zScores < 2)]

            if len(filteredValues) > 0:
                filteredMean = round(filteredValues.mean(), 6)
            else:
                filteredMean = np.nan

            # Calculate the number of outliers removed.
            outliersRemoved = len(values) - len(filteredValues)

            # Add the calculated statistics to the row.
            row["Total time Original Mean"] = originalMean
            row["Total time Filtered Mean"] = filteredMean
            row["Total time 50th Percentile"] = percentile50
            row["Total time 90th Percentile"] = percentile90
            row["Total time 99th Percentile"] = percentile99
            row["Total time Outliers Removed"] = outliersRemoved

        if "Processing Total Time" in group.columns:
            values = group["Processing Total Time"].dropna().values
            row["Processing Total Time Mean"] = round(values.mean(), 6)

        statsData.append(row)

    statsDF = pd.DataFrame(statsData)

    # Save the data to the Excel file.
    statsDF.to_excel(outputExcel, index=False)

# Specify the folder name and output name.
folderPath = "Evaluation/DLX-CPP/Colour/AllOut2"
outputExcel = "TessellationDLXCPP_ALL2_Colour_Stats.xlsx"
typeValue = "DLXCPP_Colour_All"

# Process the files and create the Excel.
processFilesToExcel(folderPath, outputExcel, typeValue)
