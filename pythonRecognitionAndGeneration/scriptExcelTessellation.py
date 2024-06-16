import os
import pandas as pd

def process_files_to_excel(folder_path, output_excel, type_value):
    data = []

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # print(f"Processing file: {file_path}")
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Extract size n from the filename
                # Assuming filename format: puzzle_nxn_piecesx_id
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

                data.append(file_data)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    # print("DataFrame head:")
    # print(df.head())  # Print the DataFrame head for debugging

    # Save the DataFrame to an Excel file
    df.to_excel(output_excel, index=False)
    # print(f"Data has been written to {output_excel}")

# Specify the folder containing the text files and the output Excel file
folder_path = "Evaluation/DLX-CPP/No-Colour"
output_excel = "TessellationDLXCPPNoColour.xlsx"
type_value = "DLXCPP_No_Colour"

# Process the files and create the Excel file
process_files_to_excel(folder_path, output_excel, type_value)
