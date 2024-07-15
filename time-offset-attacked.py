import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import correlate

# Parameters
total_time = 0.14  # Total measurement time (0.14 seconds)
avg_rate = 2000    # Average photon rate (2000 Hz)
time_bin = 100e-9    # Time bin width (1 microsecond for better resolution)

# Parameters for the Gaussian time offset distribution
true_offset = 1e-6 # True time offset (5 milliseconds)
fwhm = 135e-12     # Full Width at Half Maximum (135 picoseconds)
std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation

# Attack parameters
fixed_delay = 1e-3  # 1 nanosecond initial delay
jitter_std = 1e-11  # 10 picoseconds standard deviation for jitter
ramp_factor = 1.0001  # 0.01% increase in delay per timestamp

def gaussian(x, mu, sigma):
    return np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev):
    expected_count = int(avg_rate * total_time)
    num_photons = np.random.poisson(expected_count)
    
    arrivals_A = np.sort(np.random.uniform(0, total_time, num_photons))
    
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

def delay_attack(timestamps, fixed_delay, jitter_std, ramp_factor):
    attacked_timestamps = []
    current_delay = 0
    
    for i, timestamp in enumerate(timestamps):
        current_delay = fixed_delay * (ramp_factor ** i)
        jitter = np.random.normal(0, jitter_std)
        attacked_timestamp = timestamp + current_delay + jitter
        attacked_timestamps.append(attacked_timestamp)
    
    return np.array(attacked_timestamps)

# Generate correlated photon arrivals
arrivals_A, arrivals_B = generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev)

# Apply the attack to detector B timestamps
attacked_arrivals_B = delay_attack(arrivals_A, fixed_delay, jitter_std, ramp_factor)

# Create histograms
hist_A = create_histogram(arrivals_A, time_bin, total_time)
hist_B = create_histogram(arrivals_B, time_bin, total_time)
hist_B_attacked = create_histogram(attacked_arrivals_B, time_bin, total_time)

# Calculate cross-correlation
cross_corr, lags = calculate_cross_correlation(hist_A, hist_B)
cross_corr_attacked, lags_attacked = calculate_cross_correlation(hist_A, hist_B_attacked)

# Find the estimated offset
estimated_offset = lags[np.argmax(cross_corr)] * time_bin
estimated_offset_attacked = lags_attacked[np.argmax(cross_corr_attacked)] * time_bin

# Plot histograms
plt.figure(figsize=(12, 6))
time_axis = np.arange(0, total_time, time_bin)
plt.stairs(hist_A, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector A')
plt.stairs(hist_B, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector B (No Attack)')
plt.stairs(hist_B_attacked, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector B (Attacked)')
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.title('Correlated Photon Detection Histograms')
plt.legend()
plt.show()

# Plot cross-correlation
plt.figure(figsize=(12, 6))
plt.plot(lags * time_bin, cross_corr, label='No Attack')
plt.plot(lags_attacked * time_bin, cross_corr_attacked, label='With Attack')
plt.xlabel('Lag (s)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation of Photon Arrival Times')
plt.axvline(x=estimated_offset, color='r', linestyle='--', label='Estimated Offset (No Attack)')
plt.axvline(x=estimated_offset_attacked, color='g', linestyle='--', label='Estimated Offset (Attacked)')
plt.axvline(x=true_offset, color='k', linestyle='--', label='True Offset')
plt.legend()
plt.show()

# Calculate precision
num_runs = 100
offset_estimates = []
offset_estimates_attacked = []

for _ in range(num_runs):
    arrivals_A, arrivals_B = generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev)
    attacked_arrivals_B = delay_attack(arrivals_B, fixed_delay, jitter_std, ramp_factor)
    
    hist_A = create_histogram(arrivals_A, time_bin, total_time)
    hist_B = create_histogram(arrivals_B, time_bin, total_time)
    hist_B_attacked = create_histogram(attacked_arrivals_B, time_bin, total_time)
    
    cross_corr, lags = calculate_cross_correlation(hist_A, hist_B)
    cross_corr_attacked, lags_attacked = calculate_cross_correlation(hist_A, hist_B_attacked)
    
    offset_estimates.append(lags[np.argmax(cross_corr)] * time_bin)
    offset_estimates_attacked.append(lags_attacked[np.argmax(cross_corr_attacked)] * time_bin)

precision = np.std(offset_estimates)
precision_attacked = np.std(offset_estimates_attacked)

print(f"True offset: {true_offset:.9f} s")
print(f"Estimated offset (No Attack): {estimated_offset:.9f} s")
print(f"Estimated offset (Attacked): {estimated_offset_attacked:.9f} s")
print(f"Precision (No Attack): {precision:.9f} s")
print(f"Precision (Attacked): {precision_attacked:.9f} s")
print(f"Relative precision (No Attack): {precision/true_offset:.6f}")
print(f"Relative precision (Attacked): {precision_attacked/true_offset:.6f}")