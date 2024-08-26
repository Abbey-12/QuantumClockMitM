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

# Seed for reproducibility
np.random.seed(123)

# Parameters
total_time = 0.14  # Total measurement time in seconds
avg_rate = 2000    # Average photon rate in Hz
time_bin = 1e-4    # Time bin width in seconds

def generate_correlated_photon_arrivals(total_time, avg_rate):
    """
    Generate correlated photon arrival times.
    
    Args:
        total_time (float): Total measurement time in seconds.
        avg_rate (float): Average photon rate in Hz.

    Returns:
        np.ndarray: Array of photon arrival times.
    """
    # Expected number of events
    expected_count = avg_rate * total_time
    
    # Generate actual number of events using Poisson distribution
    num_events = np.random.poisson(expected_count)
    
    # Generate arrival times
    if num_events > 0:
        # Generate num_events - 1 inter-arrival times
        inter_arrival_times = np.random.exponential(1/avg_rate, size=num_events - 1)
        
        # Calculate arrival times
        arrivals_A = np.zeros(num_events)
        arrivals_A[1:] = np.cumsum(inter_arrival_times)
        
        # Scale arrival times to fit within total_time
        arrivals_A = arrivals_A * (total_time / arrivals_A[-1]) if arrivals_A[-1] > 0 else arrivals_A
    else:
        arrivals_A = np.array([])
    
    return arrivals_A

def create_histogram(arrivals, time_bin, total_time):
    """
    Create a histogram of photon arrival times.

    Args:
        arrivals (np.ndarray): Array of photon arrival times.
        time_bin (float): Time bin width in seconds.
        total_time (float): Total measurement time in seconds.

    Returns:
        tuple: Histogram counts and bin edges.
    """
    bins = np.arange(0, total_time + time_bin, time_bin)
    hist, _ = np.histogram(arrivals, bins=bins)
    return hist, bins

def plot_data():
    """
    Plot the histogram of photon arrival times.
    """
    # Generate photon arrivals
    arrivals_A = generate_correlated_photon_arrivals(total_time, avg_rate)
    
    # Create histogram
    hist, bins = create_histogram(arrivals_A, time_bin, total_time)
        
    # Plot histogram using stairs plot
    plt.figure(figsize=(12, 6))
    plt.stairs(hist,bins, alpha=0.5, fill=True, label='Detector 1', linewidth=2)
    
    # Adding labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Photon Count')
    plt.title('Photon Arrival Times Histogram')
    plt.legend()
    plt.show()

# Call the plot function
plot_data()
