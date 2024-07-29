from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import socket
import os
import logging
import sys
import csv
import threading

np.random.seed(123)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

total_time = 0.14  # Total measurement time (0.14 seconds)
avg_rate = 2000    # Average photon rate (2000 Hz)
time_bin = 1e-4    # Time bin width (1 microsecond for better resolution)

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

arrivals_A, arrivals_B = generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev)


# Network setup
HOST ='0.0.0.0'
# PORT = int(os.environ.get('RECEIVER_PORT', 65432))
PORT= 65432

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
        
        conn, addr = s.accept()
        logging.info(f"New connection from {addr}")
        
        buffer = ""
        received_values = []
        
        with conn:
            while True:
                chunk = conn.recv(1024).decode()
                print(chunk)
                if not chunk:
                    break
                
                buffer += chunk
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:  # Ignore empty lines
                        received_values.append(line)
        
        # print(received_values)
        logging.info(f"Connection from {addr} closed.")
        logging.info(f"Total received data points: {len(received_values)}")
        
        received_data = save_data(received_values, addr)
        return received_data, addr


def analyze_and_plot(received_data_path):
    # Read data from CSV file
    with open(received_data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        received_data = []
        for row in reader:
            try:
                value = float(row[0])
                received_data.append(value)
            except ValueError:
                logging.warning(f"Skipping invalid data point: {row[0]}")
    
    if not received_data:
        logging.error("No valid data points found in the received data.")
        return

    # Convert received data to numpy array
    received_arrivals = np.array(received_data)
    
    # Create histogram for received data
    hist_received = create_histogram(received_arrivals, time_bin, total_time)
    hist_B= create_histogram(arrivals_B,time_bin,total_time)
    # Generate correlated photon arrivals

    # Calculate cross-correlation
    cross_corr, lags = calculate_cross_correlation(hist_received, hist_B)

    # Find the estimated offset
    estimated_offset = lags[np.argmax(cross_corr)] * time_bin
    logging.info(f"Estimated offset: {estimated_offset}")
    # Plot cross-correlation
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(0, total_time, time_bin)
    plt.stairs(hist_received, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector A')
    plt.stairs(hist_B, np.arange(0, total_time + time_bin, time_bin), alpha=0.7, fill=True, label='Detector B')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.title('Correlated Photon Detection Histograms')
    plt.legend()
    # plt.show()
    plt.savefig('/data/histogram_plot.png')
    plt.close()

    # Plot cross-correlation
    plt.figure(figsize=(12, 6))
    plt.plot(lags * time_bin, cross_corr)
    plt.xlabel('Lag (s)')
    plt.ylabel('Cross-correlation')
    plt.title('Cross-correlation of Photon Arrival Times')
    plt.axvline(x=estimated_offset, color='r', linestyle='--', label='Estimated Offset')
    plt.axvline(x=true_offset, color='g', linestyle='--', label='True Offset')
    plt.legend()
    # plt.show()
    
    plt.savefig('/data/cross_correlation_plot.png')
    plt.close()

if __name__ == "__main__":
    try:
        while True:
            # np.random.seed(123)
            received_data, addr = receive_data()
            analyze_and_plot(received_data)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

    # hist_A= create_histogram(arrivals_A,time_bin, total_time)
    # hist_B= create_histogram(arrivals_B,time_bin,total_time)
    # # Calculate cross-correlation
    # cross_corr1, lags = calculate_cross_correlation(hist_A, hist_B)

    # # Find the estimated offset
    # estimated_offset1 = lags[np.argmax(cross_corr1)] * time_bin
    # logging.info(f"Estimated offset: {estimated_offset1}")
    # # Plot cross-correlation

    # # Plot cross-correlation
    # plt.figure(figsize=(12, 6))
    # plt.plot(lags * time_bin, cross_corr1)
    # plt.xlabel('Lag (s)')
    # plt.ylabel('Cross-correlation')
    # plt.title('Cross-correlation of Photon Arrival Times')
    # plt.axvline(x=estimated_offset1, color='r', linestyle='--', label='Estimated Offset')
    # plt.axvline(x=true_offset, color='g', linestyle='--', label='True Offset')
    # plt.legend()
    # plt.show()



