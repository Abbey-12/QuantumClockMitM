import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import correlate

total_time = 0.14  # Total measurement time (0.14 seconds)
avg_rate = 2000    # Average photon rate (2000 Hz)
time_bin = 1e-6   # Time bin width (1 microsecond for better resolution)

# Parameters for the Gaussian time offset distribution
true_offset = 5e-3  # True time offset (5 milliseconds)
fwhm = 135e-12         # Full Width at Half Maximum (2 milliseconds)
std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation

def gaussian(x, mu, sigma):
    return np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev):
    expected_count = int(avg_rate * total_time)
    num_photons = np.random.poisson(expected_count)
    
    arrivals_A = np.sort(np.random.uniform(0, total_time, num_photons))
    
    # Generate Gaussian-distributed offsets
    x = np.linspace(true_offset - 4*std_dev, true_offset + 4*std_dev, 1000)
    pdf = gaussian(x, true_offset, std_dev)
    offsets = np.random.choice(x, size=num_photons, p=pdf/np.sum(pdf))
    
    arrivals_B = arrivals_A + offsets
    
    arrivals_B = arrivals_B[(arrivals_B >= 0) & (arrivals_B <= total_time)]
    
    return arrivals_A, arrivals_B
def create_histogram(arrivals, time_bin, total_time):
    bins = np.arange(0, total_time + time_bin, time_bin)
    hist, _ = np.histogram(arrivals, bins=bins)
    return hist

def calculate_cross_correlation(hist_A, hist_B):
    cross_corr = correlate(hist_B, hist_A, mode='full')
    lags = np.arange(-len(hist_A) + 1, len(hist_B))
    return cross_corr, lags

# Generate correlated photon arrivals
arrivals_A, arrivals_B = generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev)

# Create histograms
hist_A = create_histogram(arrivals_A, time_bin, total_time)
hist_B = create_histogram(arrivals_B, time_bin, total_time)

# Calculate cross-correlation
cross_corr, lags = calculate_cross_correlation(hist_A, hist_B)

# Find the estimated offset
estimated_offset = lags[np.argmax(cross_corr)] * time_bin

# Plot histograms
plt.figure(figsize=(12, 6))
time_axis = np.arange(0, total_time, time_bin)
plt.stairs(hist_A, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector A')
plt.stairs(hist_B, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector B')
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.title('Correlated Photon Detection Histograms')
plt.legend()
plt.show()

# Plot cross-correlation
plt.figure(figsize=(12, 6))
plt.plot(lags * time_bin, cross_corr)
plt.xlabel('Lag (s)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation of Photon Arrival Times')
plt.axvline(x=estimated_offset, color='r', linestyle='--', label='Estimated Offset')
plt.axvline(x=true_offset, color='g', linestyle='--', label='True Offset')
plt.legend()
plt.show()

# Calculate precision
num_runs = 100
offset_estimates = []

for _ in range(num_runs):
    arrivals_A, arrivals_B = generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev)
    hist_A = create_histogram(arrivals_A, time_bin, total_time)
    hist_B = create_histogram(arrivals_B, time_bin, total_time)
    cross_corr, lags = calculate_cross_correlation(hist_A, hist_B)
    offset_estimates.append(lags[np.argmax(cross_corr)] * time_bin)

precision = np.std(offset_estimates)

print(f"True offset: {true_offset:.9f} s")
print(f"Estimated offset: {estimated_offset:.9f} s")
print(f"Precision (standard deviation of estimates): {precision:.9f} s")
print(f"Relative precision: {precision/true_offset:.6f}")