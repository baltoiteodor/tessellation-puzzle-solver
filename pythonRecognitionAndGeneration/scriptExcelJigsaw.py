import os
import pandas as pd

def process_files_to_excel(folder_path, output_excel):
    data = []

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Extract the number of pieces from the filename (last two digits before .txt)
                pieces = filename[-6:-4]

                # Create a dictionary to store the data for this file
                file_data = {"Name": filename[:-4], "Pieces": pieces}  # Remove .txt extension and add pieces

                for line in lines[1:]:  # Skip the header line
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        label, value = parts
                        file_data[label] = value

                data.append(file_data)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(output_excel, index=False)

# Specify the folder containing the text files and the output Excel file
folder_path = "Evaluation/DLX-CPP/Jigsaw"  # Replace with your folder path
output_excel = "JigsawResults.xlsx"  # Replace with your desired output file name

# Process the files and create the Excel file
process_files_to_excel(folder_path, output_excel)
