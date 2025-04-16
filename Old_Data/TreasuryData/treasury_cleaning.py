"""
This file creates one csv file for data given from treasury.gov
"""

import os
import pandas as pd

# Set the folder path
folder_path = 'Data/TreasuryData/Raw-Data'  # Update with your folder's path
output_file = 'Data/TreasuryData/Cleaned-Data/cleaned_treasury_data.csv'  # Specify the output file path

# Define the name of the date column
date_column = 'Date'  # Update with the exact column name in your data

# Initialize an empty list to store DataFrames
dataframes = []

# Iterate through all files in the folder
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    
    # Check the file extension and read the file accordingly
    if file.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file.endswith('.txt'):
        df = pd.read_csv(file_path, delimiter='\t')  # Update the delimiter if needed
    else:
        print(f"Skipping unsupported file type: {file}")
        continue
    
    # Add the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Ensure the date column is in datetime format
merged_df[date_column] = pd.to_datetime(merged_df[date_column], errors='coerce')

# Drop rows where the date conversion failed (if any)
merged_df = merged_df.dropna(subset=[date_column])

# Sort the DataFrame by date
merged_df = merged_df.sort_values(by=date_column)

# Set the date column as the index
merged_df = merged_df.set_index(date_column)

# Interpolate missing data
merged_df.interpolate(inplace=True)

# Dropping columns with missing data
merged_df.dropna(axis = 1, inplace = True)

# Dropping the 3 month treasury because it is irrelevant
merged_df.drop(columns = ["3 Mo"], inplace=True)

# Save the merged DataFrame to a single CSV file
merged_df.to_csv(output_file)

print(f"Data merged, sorted by date, and saved to {output_file}")