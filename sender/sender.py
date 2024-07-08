import numpy as np
import matplotlib.pyplot as plt
import socket
import time
import os

def master_photon_detection(detection_rate, total_time, time_bin):
    # Calculate the number of time bins
    num_bins = int(total_time / time_bin)
    
    # Calculate the expected number of photons per bin
    expected_photons_per_bin = detection_rate * time_bin
    
    #  photon detections  sequence
    detections = np.random.poisson(expected_photons_per_bin, num_bins)
    
    return detections


# Parameters
time_bin = 20e-3  # 1 millisecond bin size
detection_rate = 200  # s^-1 (assuming half of the 200 s^-1 pair rate goes to each detector)
total_time = 3 # second

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
PORT = int(os.environ.get('RECIVER_PORT', 65432))

# Check if we can resolve the hostname
ip_address = check_hostname(HOST)

if ip_address is None:
    print("Cannot resolve hostname. Please check your network configuration.")
    exit(1)

data= master_photon_detection(detection_rate, total_time, time_bin)

# print(f"Master clock generated new data: {data}")

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








