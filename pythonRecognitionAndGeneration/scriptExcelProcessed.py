import os
import pandas as pd
from scipy import stats
import numpy as np

def process_files_to_excel(folder_path, output_excel, type_value):
    data = []

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Extract size n from the filename
                parts = filename.split('_')
                size_part = parts[1]  # This is the nxn part
                size = size_part.split('x')[0]  # Extract n from nxn

                # Create a dictionary to store the data for this file
                file_data = {"Size": size, "Type": type_value}  # Use extracted size and set type

                processing_total_time = 0.0  # Initialize the total processing time

                for line in lines[1:]:  # Skip the header line
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        label, value = parts
                        if label not in ["Jigsaw Success", "Choosing the best solution for Jigsaw"]:
                            try:
                                value = round(float(value), 5)
                            except ValueError:
                                pass  # Keep the original value if it cannot be converted to float
                            file_data[label] = value

                            # Sum the specific columns into processing_total_time
                            if label in ["PreProcessing", "Contour Finding", "Rotating Pieces", "Processing Pieces into Grids"]:
                                processing_total_time += value

                # Add the new column to the dictionary
                file_data["Processing Total Time"] = round(processing_total_time, 5)

                # Remove individual preprocessing columns
                for key in ["PreProcessing", "Contour Finding", "Rotating Pieces", "Processing Pieces into Grids"]:
                    if key in file_data:
                        del file_data[key]

                data.append(file_data)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Process each column to calculate statistics
    grouped = df.groupby('Size')
    stats_data = []

    for name, group in grouped:
        row = {"Size": name, "Type": type_value}

        # Ensure there are at least 20 files in the group
        if len(group) < 20:
            fill_count = 20 - len(group)
            numeric_columns = group.select_dtypes(include=[np.number]).columns
            group_mean = group[numeric_columns].mean()
            fill_rows = [group_mean] * fill_count
            fill_df = pd.DataFrame(fill_rows)
            fill_df["Total time"] = 180
            group = pd.concat([group, fill_df], ignore_index=True)

        if "Total time" in group.columns:
            values = group["Total time"].dropna().values

            # Calculate original mean and percentiles
            original_mean = round(values.mean(), 6)
            percentile_50 = round(np.percentile(values, 50), 6)
            percentile_90 = round(np.percentile(values, 90), 6)
            percentile_99 = round(np.percentile(values, 99), 6)

            # Remove outliers using Z-score method
            z_scores = stats.zscore(values)
            filtered_values = values[(z_scores > -2) & (z_scores < 2)]

            if len(filtered_values) > 0:
                # Calculate filtered mean
                filtered_mean = round(filtered_values.mean(), 6)
            else:
                filtered_mean = np.nan

            # Calculate the number of outliers removed
            outliers_removed = len(values) - len(filtered_values)

            # Add the calculated statistics to the row
            row["Total time Original Mean"] = original_mean
            row["Total time Filtered Mean"] = filtered_mean
            row["Total time 50th Percentile"] = percentile_50
            row["Total time 90th Percentile"] = percentile_90
            row["Total time 99th Percentile"] = percentile_99
            row["Total time Outliers Removed"] = outliers_removed

        # Process the Processing Total Time separately
        if "Processing Total Time" in group.columns:
            values = group["Processing Total Time"].dropna().values
            row["Processing Total Time Mean"] = round(values.mean(), 6)

        stats_data.append(row)

    stats_df = pd.DataFrame(stats_data)

    # Save the DataFrame with statistics to an Excel file
    stats_df.to_excel(output_excel, index=False)
    print(f"Data with statistics has been written to {output_excel}")

# Specify the folder containing the text files and the output Excel file
folder_path = "Evaluation/DLX-CPP/Colour/AllOut2"  # Replace with your folder path
output_excel = "TessellationDLXCPP_ALL2_Colour_Stats.xlsx"  # Replace with your desired output file name
type_value = "DLXCPP_Colour_All"  # Replace with your desired type value

# Process the files and create the Excel file with statistics
process_files_to_excel(folder_path, output_excel, type_value)
