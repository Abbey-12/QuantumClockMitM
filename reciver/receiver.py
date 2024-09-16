from datetime import datetime
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy.signal import correlate
import numpy as np
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

total_time = 0.14
avg_rate = 2000
time_bin = 5e-6

# Parameters for the Gaussian time offset distribution
true_offset = 5e-4  # True time offset (0.5 milliseconds)
fwhm = 135e-12         # Full Width at Half Maximum (2 milliseconds)
std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
print(f"STD:{std_dev}")

DATA_DIR = '/data'
os.makedirs(DATA_DIR, exist_ok=True)

def generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev):
    # Expected number of events
    expected_count = avg_rate * total_time
    
    # Generate actual number of events using Poisson distribution
    num_photons = np.random.poisson(expected_count)
    
    # Generate num_events - 1 inter-arrival times
    inter_arrival_times = np.random.exponential(1/avg_rate, size=num_photons- 1)
    
    # Calculate arrival times
    arrivals_A = np.zeros(num_photons)
    arrivals_A[1:] = np.cumsum(inter_arrival_times)
    
    # Scale arrival times to fit within total_time
    arrivals_A = arrivals_A * (total_time / arrivals_A[-1]) if arrivals_A[-1] > 0 else arrivals_A

    offsets = np.random.normal(true_offset, std_dev, size=len(arrivals_A))
    arrivals_B = arrivals_A + offsets
    
    arrivals_B = arrivals_B[(arrivals_B >= 0) & (arrivals_B <= total_time)]

    return arrivals_A, arrivals_B

   
def save_correlation_data(cross_corr_normalized, lags):
    """
    Save the generated arrival times to CSV files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filepath_lag = os.path.join(DATA_DIR, f"Lag_arrivals{timestamp}.csv")
    # filepath_corr = os.path.join(DATA_DIR, f"corr_arrivals{timestamp}.csv")
    filepath = os.path.join(DATA_DIR, f"correlation_data_{timestamp}.csv")

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Lag (s)', 'Cross-correlation'])
        for lag, corr in zip(lags, cross_corr_normalized):
            writer.writerow([lag, corr])
    
    logging.info(f"Correlation and lag data saved to {filepath}")
    return filepath

def save_hist_data(bins, hist_A,hist_B):
    """
    Save the generated arrival times to CSV files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filepath_lag = os.path.join(DATA_DIR, f"Lag_arrivals{timestamp}.csv")
    # filepath_corr = os.path.join(DATA_DIR, f"corr_arrivals{timestamp}.csv")
    filepath = os.path.join(DATA_DIR, f"hist_data_{timestamp}.csv")

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['bins (ms)', 'hist_A','hist_B'])
        for bins, hist_A,hist_B in zip(bins, hist_A,hist_B):
            writer.writerow([bins, hist_A,hist_B])
    
    logging.info(f"Correlation and lag data saved to {filepath}")
    return filepath

def create_histogram(arrivals, time_bin, total_time):
    bins = np.arange(0, total_time + time_bin, time_bin)
    hist, _ = np.histogram(arrivals, bins=bins)
    return hist, bins

def calculate_cross_correlation(hist_A, hist_B):
    cross_corr = correlate(hist_B, hist_A, mode='full')
    lags = np.arange(-len(hist_A) + 1, len(hist_B))
        
    # Normalize the cross-correlation
    n = np.sqrt(np.sum(hist_A**2) * np.sum(hist_B**2))
    cross_corr_normalized = cross_corr / n
    save_correlation_data(cross_corr_normalized, lags)
    return cross_corr_normalized, lags

# Network setup
HOST ='0.0.0.0'
# PORT = int(os.environ.get('RECEIVER_PORT', 65432))
PORT= 65432
arrivals_A, arrivals_B = generate_correlated_photon_arrivals(total_time, avg_rate, true_offset, std_dev)
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
                chunk = conn.recv(4096).decode()
                print(chunk)
                if not chunk:
                    logging.info("No more data from client. Breaking the loop.")
                    break
                
                buffer += chunk
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:  # Ignore empty lines
                        print("print line:")
                        print(line)
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
    # Create histograms for received data and simulated data
    hist_received, bins = create_histogram(received_arrivals, time_bin, total_time)
    hist_B, bins = create_histogram(arrivals_B, time_bin, total_time)
    hist_A, bins = create_histogram(arrivals_A, time_bin, total_time)
    bins_scaled = [x * 1000 for x in bins] 
    save_hist_data(bins_scaled, hist_A,hist_B)

    # Calculate the mean of the received histogram for Poisson fitting
    mean_received = np.mean(hist_received)
    mean_B = np.mean(hist_B)

    # Generate x values for the fitted Poisson curves
    x = np.arange(0, max(max(hist_received), max(hist_B)) + 1)
    print("x")
    print(x)

    # Calculate the Poisson PMF for both histograms
    pmf_received = poisson.pmf(x, mean_received)   # Scale by bin width and number of arrivals
    pmf_B = poisson.pmf(x, mean_B)  # Scale by bin width and number of arrivals

    # Calculate cross-correlation
    cross_corr, lags = calculate_cross_correlation(hist_received, hist_B)
    lags_scaled = [x * 1000 for x in lags] 
  
    # Find the estimated offset
    estimated_offset = lags_scaled[np.argmax(cross_corr)] * time_bin
    logging.info(f"Estimated offset: {estimated_offset}")
    estimated_std = np.std(cross_corr)
    logging.info(f"Estimated std: {estimated_std}")
   
   # Define zoom window for histogram plot
    zoom_window_size = 1500  # Number of points around the peak to display for zoom
    zoom_start = max(0, np.argmax(hist_received) - zoom_window_size)
    zoom_end = min(len(hist_received), np.argmax(hist_received) + zoom_window_size)


    #  to zoom near to the peak
    peak_index = np.argmax(cross_corr)
    window_size = 100  # Number of points around the peak to display
    window_start = max(0, peak_index - window_size)
    window_end = min(len(cross_corr), peak_index + window_size)
    # Font properties
    font_properties = {'size': 14, 'weight': 'bold'}

    # Plot histograms with Poisson fits
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(0, total_time, time_bin)

    # # Plot histograms
    # plt.stairs(hist_received, bins_scaled, alpha=1, fill=True, label='Detector 1',linewidth=10)
    # plt.stairs(hist_B, bins_scaled, alpha=1, fill=True, label='Detector 2', linewidth=10)
    
    # Plot zoomed-in histogram
    plt.stairs(hist_received[zoom_start:zoom_end], bins_scaled[zoom_start:zoom_end + 1], alpha=1, fill=True, label='Detector 1', linewidth=40)
    plt.stairs(hist_B[zoom_start:zoom_end], bins_scaled[zoom_start:zoom_end + 1], alpha=1, fill=True, label='Detector 2', linewidth=40)
    # # Overlay the Poisson fits
    # plt.plot(pmf_received, x, color='red', marker='o', markersize=5, linestyle='none', label='Detector 1')
    # plt.plot(pmf_B, x, color='green', marker='o', markersize=5, linestyle='none', label='Detector 2')

    # plt.xlabel('Photon Counts per time bin', fontdict=font_properties)
    # plt.ylabel('probability', fontdict=font_properties)
    
    plt.xlabel('Time (ms)', fontdict=font_properties)
    plt.ylabel('Counts', fontdict=font_properties)
    # plt.title('Photon Detection Histograms with Poisson Fits', fontdict=font_properties)
    plt.legend(prop=font_properties)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.savefig('/data/histogram_plot_with_poisson_fits.png')
    plt.close()

    # Plot cross-correlation
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(lags_scaled[window_start:window_end]) * time_bin, cross_corr[window_start:window_end])
    # plt.plot(lags* time_bin, cross_corr)
    plt.xlabel('Lag (ms)', fontdict=font_properties)
    plt.ylabel('Cross-correlation', fontdict=font_properties)
    # plt.title('Cross-correlation of Photon Arrival Times', fontdict=font_properties)
    # plt.axvline(x=estimated_offset, color='r', linestyle='--', label=f' Normal Estimated Offset: {estimated_offset:.6f}s\n Normal Estimated Std: {estimated_std:.6f}')
    plt.text(0.05, 0.9, f'Attacked Offset: {estimated_offset:.6f}ms\nAttacked Std: {estimated_std:.6f}', 
             transform=plt.gca().transAxes, fontsize=14, color='black', weight='bold', ha='left')
    plt.legend(prop=font_properties)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
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


