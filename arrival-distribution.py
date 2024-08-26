import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import correlate

# total_time = 0.14  # Total measurement time (0.14 seconds)
# avg_rate = 2000    # Average photon rate (2000 Hz)
# time_bin = 1e-4    # Time bin width (1 microsecond for better resolution)

# # Parameters for the Gaussian time offset distribution
# true_offset = 5e-3  # True time offset (5 milliseconds)
# fwhm = 135e-12         # Full Width at Half Maximum (2 milliseconds)
# std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation

# def gaussian(x, mu, sigma):
#     return np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# def generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev):
#     expected_count = int(avg_rate * total_time)
#     num_photons = np.random.poisson(expected_count)
    
#     arrivals_A = np.sort(np.random.uniform(0, total_time, num_photons))
    
#     # Generate Gaussian-distributed offsets
#     x = np.linspace(true_offset - 4*std_dev, true_offset + 4*std_dev, 1000)
#     pdf = gaussian(x, true_offset, std_dev)
#     offsets = np.random.choice(x, size=num_photons, p=pdf/np.sum(pdf))
    
#     arrivals_B = arrivals_A + offsets
    
#     arrivals_B = arrivals_B[(arrivals_B >= 0) & (arrivals_B <= total_time)]
    
#     return arrivals_A, arrivals_B
# def create_histogram(arrivals, time_bin, total_time):
#     bins = np.arange(0, total_time + time_bin, time_bin)
#     hist, _ = np.histogram(arrivals, bins=bins)
#     return hist

# def calculate_cross_correlation(hist_A, hist_B):
#     cross_corr = correlate(hist_B, hist_A, mode='full')
#     lags = np.arange(-len(hist_A) + 1, len(hist_B))
#     return cross_corr, lags

# # Generate correlated photon arrivals
# arrivals_A, arrivals_B = generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev)

# # Create histograms
# hist_A = create_histogram(arrivals_A, time_bin, total_time)
# hist_B = create_histogram(arrivals_B, time_bin, total_time)

# # Calculate cross-correlation
# cross_corr, lags = calculate_cross_correlation(hist_A, hist_B)

# # Find the estimated offset
# estimated_offset = lags[np.argmax(cross_corr)] * time_bin

# # Plot histograms
# plt.figure(figsize=(12, 6))
# time_axis = np.arange(0, total_time, time_bin)
# plt.stairs(hist_A, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector A')
# plt.stairs(hist_B, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector B')
# plt.xlabel('Time (s)')
# plt.ylabel('Counts')
# plt.title('Correlated Photon Detection Histograms')
# plt.legend()
# plt.show()




# # Set the expected count for the Poisson distribution
# expected_count = 1
# # Number of samples to generate
# num_samples = 1000

# # Generate random photon counts from a Poisson distribution
# photon_counts = np.random.poisson(expected_count)
# print(photon_counts)

# # Create a histogram of the photon counts
# plt.figure(figsize=(10, 6))
# plt.hist(photon_counts, bins=range(0, max(photon_counts) + 1), alpha=0.75, color='blue', edgecolor='black')

# # Add titles and labels
# plt.title('Histogram of Photon Counts from Poisson Distribution')
# plt.xlabel('Number of Photons Detected')
# plt.ylabel('Frequency')
# plt.xticks(range(0, max(photon_counts) + 1))  # Set x-ticks to be integers

# # Show the plot
# plt.grid(axis='y', alpha=0.75)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameters
total_time = 0.14  # Total measurement time (0.14 seconds)
avg_rate = 2000    # Average photon rate (2000 Hz)
num_photons = np.random.poisson(int(avg_rate * total_time))  # Generate number of photons

# Generate photon arrival times
arrivals_A = np.sort(np.random.uniform(0, total_time, num_photons))

# Define time bins (e.g., 1 ms bins)
time_bin = 1e-4  # 0.1 milliseconds
bins = np.arange(0, total_time + time_bin, time_bin)  # Create bins from 0 to total_time

# Create a histogram to count arrivals in each bin
hist, bin_edges = np.histogram(arrivals_A, bins=bins)

# Fit a Poisson distribution to the histogram
# Calculate the mean of the photon counts per bin
mean_counts = np.mean(hist)
print(mean_counts)

# Generate x values for the fitted Poisson curve
x = np.arange(0, max(hist) + 1)
# Calculate the Poisson probability mass function (PMF)
pmf = poisson.pmf(x, mean_counts)  # Scale by the number of photons

# Plot the histogram
plt.figure(figsize=(12, 6))
plt.bar(bin_edges[:-1], hist, width=time_bin, alpha=0.7, color='blue', edgecolor='black', align='edge', label='Photon Arrivals')
plt.plot(x, pmf, color='red', marker='o', markersize=10, linestyle='none', label='Fitted Poisson')
plt.xlabel('Photon Counts per Bin')
plt.ylabel('Counts')
plt.title('Photon Arrival Counts with Fitted Poisson Distribution')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()