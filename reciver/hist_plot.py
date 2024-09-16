import csv
import numpy as np
import matplotlib.pyplot as plt

csv_file = "/home/abebu/SimQN/security/QuantumClockMitM/reciver/hist_data_20240902_142813.csv"

# Initialize lists to store data from each column
x_data = []
y1_data = []
y2_data = []

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    
    # Read data into lists
    for row in reader:
        if row:  # Ensure the row is not empty
            x_data.append(float(row[0]))  # First column for x-axis
            y1_data.append(float(row[1])) # Second column for y-axis 1
            y2_data.append(float(row[2])) # Third column for y-axis 2

# Convert lists to numpy arrays
x_array = np.array(x_data)
y1_array = np.array(y1_data)
y2_array = np.array(y2_data)

# Create the plot
plt.figure(figsize=(12, 6))

# Define bar width and position
bar_width = 0.35  # Width of the bars
bar1_positions = np.arange(len(x_array))
bar2_positions = bar1_positions + bar_width

# Plot the bars
plt.hist(y1_array,x_array, color='skyblue')
plt.hist(y2_array, x_array, color='orange')

# Customize the plot
plt.xlabel('X-axis (First Column)', weight='bold')
plt.ylabel('Y-axis Values', weight='bold')
plt.xticks(bar1_positions + bar_width / 2, x_array, rotation=45)  # Center x-ticks between bars

# Add legend
plt.legend()

# plt.title('Bar Plot of CSV Data', weight='bold')

# plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.show()

# Optionally, save the plot
# plt.savefig('/home/abebu/SimQN/security/QuantumClockMitM/sender/bar_plot.png', dpi=300)
