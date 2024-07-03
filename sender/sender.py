import numpy as np
import matplotlib.pyplot as plt
import socket
import time
import os

def generate_poisson_data(lambda_param, duration):
    return np.random.poisson(lambda_param, duration)

def check_hostname(hostname):
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"Hostname {hostname} resolves to IP: {ip_address}")
        return ip_address
    except socket.gaierror:
        print(f"Hostname {hostname} could not be resolved.")
        return None

# Network setup
HOST = os.environ.get('RECIVER_HOST', 'reciver')
PORT = int(os.environ.get('RECIVER_PORT', 65432))

# Check if we can resolve the hostname
ip_address = check_hostname(HOST)

if ip_address is None:
    print("Cannot resolve hostname. Please check your network configuration.")
    exit(1)

# Parameters for data generation
lambda_param = 200
duration =  100
data_size= 20000

while True:
    # Generate new data
    data = generate_poisson_data(lambda_param, data_size)
    print(f"Master clock generated new data: {data}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Attempting to connect to {ip_address}:{PORT}")
            s.connect((ip_address, PORT))
            print(f"Connected to {ip_address}:{PORT}")
            for value in data:
                s.sendall(str(value).encode() + b'\n')
                time.sleep(0.1)
            print("Data sent successfully")
    except socket.error as e:
        print(f"Connection failed: {e}")
        print("Will try again with new data in 5 seconds...")
    
    # Wait before generating and sending new data
    time.sleep(5)









# # If we couldn't connect, let's try to ping the IP
# if attempt == max_retries - 1:
#     print(f"Attempting to ping {ip_address}")
#     response = os.system(f"ping -c 1 {ip_address}")
#     if response == 0:
#         print(f"Ping to {ip_address} successful")
#     else:
#         print(f"Ping to {ip_address} failed")










# from scipy.optimize import curve_fit
# from scipy.special import voigt_profile
# # Parameters
# total_time = 100  # seconds
# pair_rate = 200  # pairs per second
# time_resolution = 4e-12  # 4 ps
# coarse_resolution = 2e-6  # 2 µs
# fine_resolution = 16e-12  # 16 ps
# fwhm = 500e-12  # 500 ps

# # Generate photon pairs
# num_pairs = np.random.poisson(pair_rate * total_time)
# pair_times = np.random.uniform(0, total_time, num_pairs)

# # Simulate detection and time-stamping
# def add_jitter(times):
#     return times + np.random.normal(0, fwhm / (2 * np.sqrt(2 * np.log(2))), len(times))

# alice_times = add_jitter(pair_times)
# bob_times = add_jitter(pair_times + np.random.normal(0, 1e-9, num_pairs))  # Add some delay variation

# # Calculate time differences
# time_diffs = bob_times - alice_times

# # Coincidence counting function
# def count_coincidences(diffs, resolution):
#     bins = np.arange(min(diffs), max(diffs) + resolution, resolution)
#     counts, _ = np.histogram(diffs, bins)
#     return bins[:-1] + resolution/2, counts

# # Get coincidence counts for coarse and fine resolutions
# coarse_bins, coarse_counts = count_coincidences(time_diffs, coarse_resolution)
# fine_bins, fine_counts = count_coincidences(time_diffs, fine_resolution)

# # Corrected Pseudo-Voigt profile
# def pseudo_voigt(x, amp, center, sigma, f):
#     return amp * ((1-f) * np.exp(-(x-center)**2 / (2*sigma**2)) + 
#                   f * voigt_profile(x-center, sigma, 0))

# # Fit function
# def fit_func(x, a0, a1, a2, tau_AB, tau_BA):
#     return (a0 + 
#             a1 * pseudo_voigt(x, 1, tau_AB, 290e-12, 0.2) + 
#             a2 * pseudo_voigt(x, 1, tau_BA, 290e-12, 0.2))

# # Fit the fine resolution data
# popt, _ = curve_fit(fit_func, fine_bins, fine_counts, 
#                     p0=[min(fine_counts), max(fine_counts), max(fine_counts), 0, 1e-9])

# # Plot results
# plt.figure(figsize=(12, 8))
# plt.plot(fine_bins, fine_counts, 'b.', label='Data')
# plt.plot(fine_bins, fit_func(fine_bins, *popt), 'r-', label='Fit')
# plt.xlabel('Time difference (s)')
# plt.ylabel('Coincidence counts')
# plt.legend()
# plt.title('Coincidence Peak Fitting')
# plt.show()

# # Print results
# print(f"Background coincidences (a0): {popt[0]:.2f}")
# print(f"Detected pairs (a1, a2): {popt[1]:.2f}, {popt[2]:.2f}")
# print(f"Peak positions (τAB, τBA): {popt[3]*1e9:.2f} ns, {popt[4]*1e9:.2f} ns")
# print(f"Time offset (δ): {(popt[4] - popt[3])*1e9/2:.2f} ns")
# print(f"Time difference (ΔT): {(popt[4] + popt[3])*1e9/2:.2f} ns")