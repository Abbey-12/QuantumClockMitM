import csv
import matplotlib.pyplot as plt
import numpy as np

time_bin = 5e-6

# Function to read CSV data
def read_csv(file_path):
    x_data = []
    y_data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header if there is one
        for row in reader:
            x_data.append(float(row[0]))  # Convert first column to float
            y_data.append(float(row[1]))  # Convert second column to float
    return x_data, y_data

# Read the CSV files
x1, y1 = read_csv('/home/abebu/SimQN/security/QuantumClockMitM/reciver/correlation_data_20240827_093550.csv')
x2, y2 = read_csv('/home/abebu/SimQN/security/QuantumClockMitM/reciver/correlation_data_20240827_093619.csv')

# Find the estimated offset
estimated_offset_normal = x1[np.argmax(y1)] * time_bin*1000
estimated_std_normal = np.std(y1)

# Find the estimated offset
estimated_offset_attacked = x2[np.argmax(y2)] * time_bin*1000
estimated_std_attacked = np.std(y2)

# to zoom near to the peak
peak_index1 = np.argmax(y1)
window_size = 30 # Number of points around the peak to display
window_start = max(0, peak_index1 - window_size)
window_end = min(len(y1), peak_index1 + window_size)
lags_scaled1 = [x * 1000 for x in x1] 

# to zoom near to the peak
peak_index2 = np.argmax(y2)
window_size = 30  # Number of points around the peak to display
window_start = max(0, peak_index2 - window_size)
window_end = min(len(y2), peak_index2 + window_size)
lags_scaled2 = [x * 1000 for x in x2] 


# Plotting the data
plt.figure(figsize=(16, 9))
font_properties = {'size': 25, 'weight': 'bold'}

# Make the spines bold by increasing their linewidth
ax = plt.gca()  # Get the current axis
for spine in ax.spines.values():
    spine.set_linewidth(2)  # Set the linewidth (thicker border)

plt.plot(np.array(lags_scaled1[window_start:window_end]) * 1e-4, y1[window_start:window_end], 
         label=f'Normal Offset: {estimated_offset_normal:.6f}ms\nNormal Std: {estimated_std_normal:.6f}', color='blue')
plt.plot(np.array(lags_scaled2[window_start:window_end]) * 1e-4, y2[window_start:window_end], 
         label=f'Attacked Offset: {estimated_offset_attacked:.6f}ms\nAttacked Std: {estimated_std_attacked:.6f}', color='red')

# Adding titles and labels
plt.xlabel('Lag (ms)', fontdict=font_properties)
plt.ylabel('Normalized Correlation', fontdict=font_properties)
plt.legend(fontsize=24, loc='center left', bbox_to_anchor=(0, 0.8)) 
plt.tick_params(labelsize = 20)
# Save the plot as a PDF file
plt.savefig('correlation_plot.pdf',dpi=300)

# Show the plot
plt.show()