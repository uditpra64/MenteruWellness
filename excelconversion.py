import os
import pandas as pd
import glob

# Path to the folder containing all the files
folder_path = r"C:\Users\uditp\Takenaka-Wellness\00_Data\03_Output"

# Get all Excel files in the folder
excel_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Count for tracking progress
total_files = len(excel_files)
processed_files = 0

# Process each file
for file_path in excel_files:
    try:
        # Extract the filename without path
        file_name = os.path.basename(file_path)
        
        # Create output file name (add _utf8 suffix)
        output_file = os.path.join(folder_path, file_name.replace(".csv", "_utf8.csv"))
        
        # Read with cp932 encoding
        df = pd.read_csv(file_path, encoding="cp932")
        
        # Write with UTF-8 encoding
        df.to_csv(output_file, encoding="utf-8-sig", index=False)
        
        # Update count and print progress
        processed_files += 1
        print(f"[{processed_files}/{total_files}] Converted: {file_name}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"\nCompleted! {processed_files} out of {total_files} files were successfully converted.")