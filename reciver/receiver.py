from datetime import datetime
from scipy.stats import poisson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import socket
import os
import logging
import sys
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def master_photon_detection(detection_rate, total_time, time_bin):
    num_bins = int(total_time / time_bin)
    expected_photons_per_bin = detection_rate * time_bin
    detections = np.random.poisson(expected_photons_per_bin, num_bins)
    return detections

def slave_photon_detection(first_detections, delay_bins, noise_level):
    second_detections = np.roll(first_detections, delay_bins)
    noise = np.random.normal(0, noise_level, size=second_detections.shape)
    second_detections = np.clip(second_detections + noise, 0, None).astype(int)
    return second_detections

# Parameters
time_bin = 20e-3  # 20 milliseconds bin size
detection_rate = 200  # s^-1
total_time = 3  # seconds
correlation_delay = 1e-2  # 10 milliseconds
noise_level = 1  # Standard deviation of the noise added to the second detections

photon_counts_1 = master_photon_detection(detection_rate, total_time, time_bin)
delay_bins = int(correlation_delay / time_bin)
photon_counts_2 = slave_photon_detection(photon_counts_1, delay_bins, noise_level)
num_bins = len(photon_counts_1)
time_array = np.linspace(0, total_time, num_bins, endpoint=False)

# Network setup
HOST = '0.0.0.0'
PORT = int(os.environ.get('RECEIVER_PORT', 65432))

DATA_DIR = '/data'
os.makedirs(DATA_DIR, exist_ok=True)

def save_data(data, source_addr):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"received_data_{timestamp}_{source_addr[0]}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Value'])
        for value in data:
            writer.writerow([value])
    logging.info(f"Data saved to {filepath}")
    return filepath

def receive_data():
    logging.info(f"Receiver starting up on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        logging.info(f"Receiver listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            logging.info(f"New connection from {addr}")
            received_data = []
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    received_values = data.decode().strip().split('\n')
                    received_data.extend(received_values)
                    for value in received_values:
                        logging.info(f"Received value: {value}")
            logging.info(f"Connection from {addr} closed.")
            logging.info(f"Total received data points: {len(received_data)}")
            received_data_file = save_data(received_data, addr)
            correlate_data(photon_counts_2, received_data_file)

def correlate_data(photon_counts_2, received_data_file):
    df = pd.read_csv(received_data_file)
    received_data = df['Value'].astype(int).values
    cross_corr = np.correlate(received_data, photon_counts_2, mode='full')
    lags = np.arange(-num_bins + 1, num_bins)
    normalization_factor = np.sqrt(np.sum(received_data**2) * np.sum(photon_counts_2**2))
    normalized_cross_corr = cross_corr / normalization_factor
    max_corr_index = np.argmax(normalized_cross_corr)
    max_corr_lag = lags[max_corr_index] * time_bin
    print(f"Maximum correlation at lag: {max_corr_lag:.3f} seconds")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

    # Plot 1: Comparison of received data and locally generated data
    print("Difference between the received and locally generated: ", np.mean(received_data - photon_counts_1))

    ax1.step(time_array, received_data, where='post', label='Received data', alpha=0.7)
    ax1.step(time_array, photon_counts_1, where='post', label='Local data', alpha=0.7, linestyle='--')
    ax1.set_xlim(0, total_time)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Photon Counts per bin')
    ax1.set_title('Comparison of Received and Local Data')
    ax1.legend()

    # Plot 2: Photon Detection Sequence
    ax2.step(time_array, received_data, where='post', label='Received detections')
    ax2.step(time_array, photon_counts_2, where='post', label='Second detections', linestyle='--')
    ax2.set_xlim(0, total_time)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Photon Counts per bin')
    ax2.set_title('Photon Detection Sequence')
    ax2.legend()

    # Plot 3: Distribution of Photon Counts
    unique, counts = np.unique(received_data, return_counts=True)
    ax3.bar(unique, counts / len(received_data), alpha=0.7, label='Received data')
    ax3.set_xlabel('Photon Counts per bin')
    ax3.set_ylabel('Probability')
    ax3.set_title('Distribution of Photon Counts (Received data)')
    mean_counts = np.mean(received_data)
    x = np.arange(0, max(received_data) + 1)
    poisson_fit = poisson.pmf(x, mean_counts)
    ax3.plot(x, poisson_fit, 'r-', lw=2, label='Theoretical Poisson')
    ax3.legend()

    # Plot 4: Normalized Cross-correlation
    ax4.plot(lags * time_bin, normalized_cross_corr)
    ax4.set_xlabel('Lag (s)')
    ax4.set_ylabel('Normalized Cross-correlation')
    ax4.set_title('Normalized Cross-correlation between Received and Second Detections')
    ax4.axvline(correlation_delay, color='r', linestyle='--', label='Expected delay')
    ax4.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        receive_data()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)