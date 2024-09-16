import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

csv_file = "/home/abebu/SimQN/security/QuantumClockMitM/sender/20240831_143235.csv"

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    data = [float(row[0]) for row in reader if row]  # Read all rows, converting to float

# Convert to numpy array
array_data = np.array(data)

# Calculate the probability density function
kde = stats.gaussian_kde(array_data)
x_range = np.linspace(array_data.min(), array_data.max(), 1000)
pdf = kde(x_range)


# Create the plot
plt.figure(figsize=(15, 9))
font_properties = {'size': 25, 'weight': 'bold'}
# Make the spines bold by increasing their linewidth
ax = plt.gca()  # Get the current axis
for spine in ax.spines.values():
    spine.set_linewidth(2)  # Set the linewidth (thicker border)

# Plot the histogram
plt.hist(array_data, bins=50, density=True, alpha=1, color='skyblue', edgecolor='black')

# Plot the PDF
plt.plot(x_range, pdf, color='red', linewidth=2)

# Set labels with bold weight
plt.xlabel('Number of Photon Counts', weight='bold',fontsize = 25 )
plt.ylabel('Probability', weight='bold', fontsize = 25 )
plt.tick_params(labelsize = 20)
# Optionally, add legend
plt.legend(['PDF', 'Histogram'], fontsize=25)
plt.savefig('/home/abebu/SimQN/security/QuantumClockMitM/sender/probability_distribution.pdf', dpi=300)
# Show the plot
plt.show()



# # This is the example of reading the .csv file and plot it to the IEEE Future networks forum
# # print(dataset.binms)
# plt.figure(figsize=(12,6))
# plt.bar(dataset.binsms[1:1200], dataset.hist_A[1:1200], width=0.05)
# plt.bar(dataset.binsms[1:1200], dataset.hist_B[1:1200], width=0.05)
# plt.legend(['Detector A','Detector B'], fontsize = 25)
# plt.xlabel('Time (ms)', weight='bold', fontsize = 25)
# plt.ylabel('Counts', weight='bold', fontsize = 25)
# plt.tick_params(labelsize = 20)
# plt.savefig('photon_arrival.pdf',bbox_inches = 'tight', dpi=300)
# plt.show()