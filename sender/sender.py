
from scipy.signal import correlate
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import socket
import time
import os
import csv

# np.random.seed(123)

total_time = 0.14
avg_rate = 2000
time_bin = 1e-4

def generate_correlated_photon_arrivals(total_time, avg_rate):
    expected_count = int(avg_rate * total_time)
    num_photons = np.random.poisson(expected_count)
    arrivals_A = np.sort(np.random.uniform(0, total_time, num_photons)) 
    return arrivals_A

def check_hostname(hostname):
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"Hostname {hostname} resolves to IP: {ip_address}")
        return ip_address
    except socket.gaierror:
        print(f"Hostname {hostname} could not be resolved.")
        return None

def save_arrivals(arrivals, filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for arrival in arrivals:
            writer.writerow([f"{arrival:.9f}"])

HOST = os.environ.get('RECEIVER_HOST', 'slave')
PORT = 65432

ip_address = check_hostname(HOST)

if ip_address is None:
    print("Cannot resolve hostname. Please check your network configuration.")
    exit(1)

def send_data():
    # try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Attempting to connect to {ip_address}:{PORT}")
        s.connect((ip_address, PORT))
        print(f"Connected to {ip_address}:{PORT}")
        
        # while True:
        arrivals_A = generate_correlated_photon_arrivals(total_time, avg_rate)
        
        # Save arrivals to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arrivals_{timestamp}.csv"
        save_arrivals(arrivals_A, filename)
        print(f"Saved {len(arrivals_A)} data points to {filename}")
        
        for value in arrivals_A:
            s.sendall(f"{value:.9f}\n".encode())
        print(f"Sent {len(arrivals_A)} data points")

        
while True:
# for i in range(4):
    np.random.seed(123)
    send_data()
    time.sleep(10)
    # print("new session")


