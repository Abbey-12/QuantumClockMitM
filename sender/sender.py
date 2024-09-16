
from scipy.signal import correlate
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import poisson
import numpy as np
import socket
import time
import os
import csv


total_time = 0.14
avg_rate = 2000
time_bin = 5e-6

def generate_correlated_photon_arrivals(total_time, avg_rate):
    # Expected number of events
    expected_count = avg_rate * total_time
    
    # Generate actual number of events using Poisson distribution
    # num_events_pdf = np.random.poisson(expected_count, size=1000)
    num_events = np.random.poisson(expected_count)
    # Generate num_events - 1 inter-arrival times
    inter_arrival_times = np.random.exponential(1/avg_rate, size=num_events - 1)
    
    # Calculate arrival times
    arrivals_A = np.zeros(num_events)
    arrivals_A[1:] = np.cumsum(inter_arrival_times)
    
    # Scale arrival times to fit within total_time
    arrivals_A = arrivals_A * (total_time / arrivals_A[-1]) if arrivals_A[-1] > 0 else arrivals_A
    return arrivals_A


def check_hostname(hostname):
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"Hostname {hostname} resolves to IP: {ip_address}")
        return ip_address
    except socket.gaierror:
        print(f"Hostname {hostname} could not be resolved.")
        return None

def save_arrivals(number_events_pdf, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Arrival Time', 'Index'])  # Write header
        for events in number_events_pdf:
            writer.writerow([f"{events:.9f}"])

HOST = os.environ.get('RECEIVER_HOST', 'slave')
PORT = 65432

ip_address = check_hostname(HOST)

if ip_address is None:
    print("Cannot resolve hostname. Please check your network configuration.")
    exit(1)


def create_histogram(arrivals, time_bin, total_time):
    bins = np.arange(0, total_time + time_bin, time_bin)
    hist, _ = np.histogram(arrivals, bins=bins)
    return hist,bins

def send_data():
    # try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Attempting to connect to {ip_address}:{PORT}")
        s.connect((ip_address, PORT))
        print(f"Connected to {ip_address}:{PORT}")
        
        # while True:
        arrivals_A = generate_correlated_photon_arrivals(total_time, avg_rate)
        for value in arrivals_A:
            s.sendall(f"{value:.9f}\n".encode())
        print(f"Sent {len(arrivals_A)} data points")
    
    # # Save arrivals to a file
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{timestamp}.csv"
    # save_arrivals(number_events_pdf, filename)
    # print(f"Saved {len(arrivals_A)} data points to {filename}")
           
while True:
    np.random.seed(123)
    send_data()
    time.sleep(1)
 

