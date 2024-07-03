import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# def simulate_photon_detection(detection_rate, total_time, time_bin):
#     # Calculate the number of time bins
#     num_bins = int(total_time / time_bin)
    
#     # Calculate the expected number of photons per bin
#     expected_photons_per_bin = detection_rate * time_bin
    
#     # Simulate photon detections
#     detections = np.random.poisson(expected_photons_per_bin, num_bins)
    
#     return detections

# def generate_correlated_detection(first_detections, delay_bins):
#     # Shift the first detections by the delay to create correlated second detections
#     second_detections = np.roll(first_detections, delay_bins)
#     return second_detections

# # Parameters
# time_bin = 1e-4  # 1 millisecond bin size
# detection_rate = 200  # s^-1 (assuming half of the 200 s^-1 pair rate goes to each detector)
# total_time = 1  # second
# correlation_delay = 0.01  # 10 milliseconds

# # Run simulation
# photon_counts_1 = simulate_photon_detection(detection_rate, total_time, time_bin)

# # Calculate the number of bins corresponding to the correlation delay
# delay_bins = int(correlation_delay / time_bin)

# # Generate second correlated detection event
# photon_counts_2 = generate_correlated_detection(photon_counts_1, delay_bins)

# # Calculate time array
# num_bins = len(photon_counts_1)
# time_array = np.linspace(0, total_time, num_bins, endpoint=False)

# # Calculate cross-correlation
# cross_corr = np.correlate(photon_counts_1, photon_counts_2, mode='full')
# lags = np.arange(-num_bins + 1, num_bins)

# # Normalize the cross-correlation
# normalization_factor = np.sqrt(np.sum(photon_counts_1**2) * np.sum(photon_counts_2**2))
# normalized_cross_corr = cross_corr / normalization_factor

# # Plotting
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# # Time sequence plot for first and second detections
# ax1.step(time_array, photon_counts_1, where='post', label='First detections')
# ax1.step(time_array, photon_counts_2, where='post', label='Second detections', linestyle='--')
# ax1.set_xlim(0, total_time)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Photon Counts per bin')
# ax1.set_title('Photon Detection Sequence')
# ax1.legend()

# # Histogram of photon counts for first detections
# unique, counts = np.unique(photon_counts_1, return_counts=True)
# ax2.bar(unique, counts / len(photon_counts_1), alpha=0.7, label='Simulated data (First detections)')
# ax2.set_xlabel('Photon Counts per bin')
# ax2.set_ylabel('Probability')
# ax2.set_title('Distribution of Photon Counts (First detections)')

# # Theoretical Poisson distribution for first detections
# mean_counts = np.mean(photon_counts_1)
# x = np.arange(0, max(photon_counts_1) + 1)
# poisson_fit = poisson.pmf(x, mean_counts)
# ax2.plot(x, poisson_fit, 'r-', lw=2, label='Theoretical Poisson')
# ax2.legend()

# # Normalized cross-correlation plot
# ax3.plot(lags * time_bin, normalized_cross_corr)
# ax3.set_xlabel('Lag (s)')
# ax3.set_ylabel('Normalized Cross-correlation')
# ax3.set_title('Normalized Cross-correlation between First and Second Detections')
# ax3.axvline(correlation_delay, color='r', linestyle='--', label='Expected delay')
# ax3.legend()

# plt.tight_layout()
# plt.show()

# print(f"Total number of detected photons (first detections): {np.sum(photon_counts_1)}")
# print(f"Total number of detected photons (second detections): {np.sum(photon_counts_2)}")
# print(f"Average detection rate (first detections): {np.sum(photon_counts_1)/total_time:.2f} s^-1")
# print(f"Average detection rate (second detections): {np.sum(photon_counts_2)/total_time:.2f} s^-1")
# print(f"Mean photon count per bin (first detections): {mean_counts:.2f}")
# print(f"Mean photon count per bin (second detections): {np.mean(photon_counts_2):.2f}")

# # Check the maximum correlation lag
# max_corr_index = np.argmax(normalized_cross_corr)
# max_corr_lag = lags[max_corr_index] * time_bin
# print(f"Maximum correlation at lag: {max_corr_lag:.3f} seconds")




def simulate_photon_detection(detection_rate, total_time, time_bin):
    # Calculate the number of time bins
    num_bins = int(total_time / time_bin)
    
    # Calculate the expected number of photons per bin
    expected_photons_per_bin = detection_rate * time_bin
    
    # Simulate photon detections
    detections = np.random.poisson(expected_photons_per_bin, num_bins)
    
    return detections

def generate_correlated_detection_with_noise(first_detections, delay_bins, noise_level):
    # Shift the first detections by the delay to create correlated second detections
    second_detections = np.roll(first_detections, delay_bins)
    
    # Add noise
    noise = np.random.normal(0, noise_level, size=second_detections.shape)
    second_detections = np.clip(second_detections + noise, 0, None).astype(int)
    
    return second_detections

# Parameters
time_bin = 20e-3  # 1 millisecond bin size
detection_rate = 200  # s^-1 (assuming half of the 200 s^-1 pair rate goes to each detector)
total_time = 3 # second
correlation_delay = 1e-5 # 10 milliseconds
noise_level = 1  # Standard deviation of the noise added to the second detections

# Run simulation
photon_counts_1 = simulate_photon_detection(detection_rate, total_time, time_bin)

# Calculate the number of bins corresponding to the correlation delay
delay_bins = int(correlation_delay / time_bin)

# Generate second correlated detection event with noise
photon_counts_2 = generate_correlated_detection_with_noise(photon_counts_1, delay_bins, noise_level)

# Calculate time array
num_bins = len(photon_counts_1)
time_array = np.linspace(0, total_time, num_bins, endpoint=False)

# Calculate cross-correlation
cross_corr = np.correlate(photon_counts_1, photon_counts_2, mode='full')
lags = np.arange(-num_bins + 1, num_bins)

# Normalize the cross-correlation
normalization_factor = np.sqrt(np.sum(photon_counts_1**2) * np.sum(photon_counts_2**2))
normalized_cross_corr = cross_corr / normalization_factor

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# Time sequence plot for first and second detections
ax1.step(time_array, photon_counts_1, where='post', label='First detections')
ax1.step(time_array, photon_counts_2, where='post', label='Second detections', linestyle='--')
ax1.set_xlim(0, total_time)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Photon Counts per bin')
ax1.set_title('Photon Detection Sequence')
ax1.legend()

# Histogram of photon counts for first detections
unique, counts = np.unique(photon_counts_1, return_counts=True)
ax2.bar(unique, counts / len(photon_counts_1), alpha=0.7, label='Simulated data (First detections)')
ax2.set_xlabel('Photon Counts per bin')
ax2.set_ylabel('Probability')
ax2.set_title('Distribution of Photon Counts (First detections)')

# Theoretical Poisson distribution for first detections
mean_counts = np.mean(photon_counts_1)
x = np.arange(0, max(photon_counts_1) + 1)
poisson_fit = poisson.pmf(x, mean_counts)
ax2.plot(x, poisson_fit, 'r-', lw=2, label='Theoretical Poisson')
ax2.legend()

# Normalized cross-correlation plot
ax3.plot(lags * time_bin, normalized_cross_corr)
ax3.set_xlabel('Lag (s)')
ax3.set_ylabel('Normalized Cross-correlation')
ax3.set_title('Normalized Cross-correlation between First and Second Detections')
ax3.axvline(correlation_delay, color='r', linestyle='--', label='Expected delay')
ax3.legend()

# plt.tight_layout()
# plt.show()

# Print information
print(f"Total number of detected photons (first detections): {np.sum(photon_counts_1)}")
print(f"Total number of detected photons (second detections): {np.sum(photon_counts_2)}")
print(f"Average detection rate (first detections): {np.sum(photon_counts_1)/total_time:.2f} s^-1")
print(f"Average detection rate (second detections): {np.sum(photon_counts_2)/total_time:.2f} s^-1")
print(f"Mean photon count per bin (first detections): {mean_counts:.2f}")
print(f"Mean photon count per bin (second detections): {np.mean(photon_counts_2):.2f}")

# Check the maximum correlation lag
max_corr_index = np.argmax(normalized_cross_corr)
max_corr_lag = lags[max_corr_index] * time_bin
print(f"Maximum correlation at lag: {max_corr_lag:.3f} seconds")
