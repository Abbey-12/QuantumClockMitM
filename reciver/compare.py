import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# # File path
# file_path1 = "/home/abebu/SimQN/security/QuantumClockMitM/reciver/data/received_data_20240715_102514_127.0.0.1.csv"
# file_path2 = "/home/abebu/SimQN/security/QuantumClockMitM/sender/data/arrivals_A.csv"

# # Read the CSV file
# df = pd.read_csv(file_path)

# # Assuming the first column contains the time data
# column_to_plot = df.columns[0]

# # Total time and bin size
# total_time = 0.14  # 0.14 seconds
# bin_size = 1e-3   # 1 microsecond

# # Calculate the number of bins
# num_bins = int(total_time / bin_size)

# # Create the histogram
# hist, bin_edges = np.histogram(df[column_to_plot], bins=num_bins, range=(0, total_time))

# # Calculate bin centers for plotting
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# # Create the plot
# plt.figure(figsize=(12, 6))

# # Plot the histogram
# plt.bar(bin_centers, hist, width=bin_size, alpha=0.7, color='skyblue', edgecolor='black')

# # Customize the plot
# plt.title(f'Histogram of {column_to_plot}')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Count')
# plt.grid(True, alpha=0.3)

# # Set x-axis limits
# plt.xlim(0, total_time)

# # Add text about bin size
# plt.text(0.98, 0.95, f'Bin size: {bin_size} s', 
#          horizontalalignment='right', verticalalignment='top', 
#          transform=plt.gca().transAxes)

# # Show the plot
# plt.show()


# File paths
file_path1 = "/home/abebu/SimQN/security/QuantumClockMitM/reciver/data/received_data_20240715_105026_127.0.0.1.csv"
file_path2 = "/home/abebu/SimQN/security/QuantumClockMitM/sender/data/arrivals_A.csv"

# Read the CSV files
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Assuming the first column contains the data we want to compare in both files
column_to_compare1 = df1.columns[0]
column_to_compare2 = df2.columns[0]

# Ensure both dataframes have the same number of rows
min_rows = min(len(df1), len(df2))
df1 = df1.head(min_rows)
df2 = df2.head(min_rows)

# Compare values
equal_values = np.isclose(df1[column_to_compare1], df2[column_to_compare2], rtol=1e-5, atol=1e-8)

# Calculate statistics
total_values = len(equal_values)
matching_values = np.sum(equal_values)
non_matching_values = total_values - matching_values

print(f"Total values compared: {total_values}")
print(f"Matching values: {matching_values}")
print(f"Non-matching values: {non_matching_values}")
print(f"Percentage of matching values: {(matching_values/total_values)*100:.2f}%")

# If you want to see the non-matching values:
if non_matching_values > 0:
    print("\nNon-matching values (first 10):")
    non_matching_indices = np.where(~equal_values)[0][:10]
    for idx in non_matching_indices:
        print(f"Index {idx}: File1 = {df1[column_to_compare1].iloc[idx]}, File2 = {df2[column_to_compare2].iloc[idx]}")