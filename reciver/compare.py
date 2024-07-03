import csv
import sys

# File paths should be strings
file1_path = "/home/abebu/SimQN/security/reciver/data/received_data_20240702_092817_172.19.0.4.csv"
file2_path = "/home/abebu/SimQN/security/reciver/data/received_data_20240702_073952_172.19.0.3.csv"

def compare_csv_files(file1_path, file2_path):
    # Read the contents of both CSV files
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        csv1 = list(csv.reader(file1))
        csv2 = list(csv.reader(file2))

    # Check if the files have the same number of rows
    if len(csv1) != len(csv2):
        print(f"Files have different number of rows: {len(csv1)} vs {len(csv2)}")
        return

    # Compare the contents of both files
    differences = []
    for row_num, (row1, row2) in enumerate(zip(csv1, csv2), start=1):
        if row1 != row2:
            differences.append(f"Row {row_num}: {row1} != {row2}")

    # Print the results
    if differences:
        print("Differences found:")
        for diff in differences:
            print(diff)
    else:
        print("No differences found. The files are identical.")

if __name__ == "__main__":
    # Use the predefined file paths instead of command-line arguments
    compare_csv_files(file1_path, file2_path)