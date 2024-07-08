import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# File path
file_path = "/home/abebu/SimQN/security/reciver/data/received_data_20240703_151343_172.19.0.3.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Assuming you want to plot the first column. Adjust this if needed.
column_to_plot = df.columns[0]

# Get value counts (which gives unique values and their frequencies)
value_counts = df[column_to_plot].value_counts().sort_index()

# Calculate the mean of the data (lambda for Poisson distribution)
lambda_param = df[column_to_plot].mean()

# Create a range of x values for the Poisson distribution
x = np.arange(0, max(value_counts.index) + 1)

# Calculate the Poisson probabilities
poisson_prob = poisson.pmf(x, lambda_param) * len(df)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the histogram
plt.bar(value_counts.index, value_counts.values, alpha=0.7, label='Observed')

# Plot the Poisson distribution
plt.plot(x, poisson_prob, 'r-', label='Poisson Distribution')

plt.title(f'Histogram of {column_to_plot} with Poisson Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig('histogram_with_poisson.png')

# Display the plot (if you're running this in an environment that can show plots)
plt.show()

# Print the value counts and Poisson parameters
print("Observed Value Counts:")
print(value_counts)
print(f"\nPoisson Distribution Parameter (lambda): {lambda_param}")