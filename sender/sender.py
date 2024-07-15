from scipy.signal import correlate
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import socket
import time
import os
import csv

np.random.seed(123)
total_time = 0.14  # Total measurement time (0.14 seconds)
avg_rate = 2000    # Average photon rate (2000 Hz)
time_bin = 1e-4  # Time bin width (1 microsecond for better resolution)

def generate_correlated_photon_arrivals(total_time, avg_rate):
    expected_count = int(avg_rate * total_time)
    num_photons = np.random.poisson(expected_count)
    
    arrivals_A = np.sort(np.random.uniform(0, total_time, num_photons)) 
    return arrivals_A

def create_histogram(arrivals, time_bin, total_time):
    bins = np.arange(0, total_time + time_bin, time_bin)
    hist, _ = np.histogram(arrivals, bins=bins)
    return hist


# Generate correlated photon arrivals
arrivals_A = generate_correlated_photon_arrivals(total_time, avg_rate)

print(len(arrivals_A))

# DATA_DIR = '/home/abebu/SimQN/security/QuantumClockMitM/sender/data'
# os.makedirs(DATA_DIR, exist_ok=True)

# # Save arrivals_A to a CSV file in the specified directory
# filename = os.path.join(DATA_DIR, 'arrivals_A.csv')
# with open(filename, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Arrival Time'])  # Header
#     for arrival in arrivals_A:
#         writer.writerow([f'{arrival:.9f}'])
# print(f"Saved arrivals_A to {filename}")

# Create histograms
hist_A = create_histogram(arrivals_A, time_bin, total_time)

# # Plot histograms
# plt.figure(figsize=(12, 6))
# time_axis = np.arange(0, total_time, time_bin)
# plt.stairs(hist_A, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector A')
# plt.xlabel('Time (s)')
# plt.ylabel('Counts')
# plt.title('Correlated Photon Detection Histograms')
# plt.legend()
# plt.show()

# Network setup
def check_hostname(hostname):
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"Hostname {hostname} resolves to IP: {ip_address}")
        return ip_address
    except socket.gaierror:
        print(f"Hostname {hostname} could not be resolved.")
        return None


HOST = os.environ.get('RECIVER_HOST', 'slave')
# HOST = '127.0.0.1'
PORT = 65432

# Check if we can resolve the hostname
ip_address = check_hostname(HOST)

if ip_address is None:
    print("Cannot resolve hostname. Please check your network configuration.")
    exit(1)

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Attempting to connect to {ip_address}:{PORT}")
        s.connect((ip_address, PORT))
        print(f"Connected to {ip_address}:{PORT}")
        print(str(arrivals_A))
        for value in arrivals_A:
            print(str(value))
            s.sendall(str(value).encode() + b'\n')
            print(str(value))
            time.sleep(0.1)
        print("Data sent successfully")
except socket.error as e:
    print(f"Connection failed: {e}")
    print("Will try again with new data in 5 seconds...")

# Wait before generating and sending new data
time.sleep(5)