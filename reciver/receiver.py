
import socket
import os
import logging
import sys
import csv
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Network setup
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = int(os.environ.get('RECEIVER_PORT', 65432))  # Get port from environment variable or use default

# Data storage setup
DATA_DIR = '/data'  # Assuming you'll mount a volume here in Docker
os.makedirs(DATA_DIR, exist_ok=True)

def save_data(data, source_addr):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"received_data_{timestamp}_{source_addr[0]}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Value'])  # Header
        for value in data:
            writer.writerow([value])
    
    logging.info(f"Data saved to {filepath}")

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
            
            # Save the received data
            save_data(received_data, addr)

if __name__ == "__main__":
    try:
        receive_data()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)



























# import socket
# import os
# import logging
# import sys

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# # Network setup
# HOST = '0.0.0.0'  # Listen on all available interfaces
# PORT = int(os.environ.get('RECEIVER_PORT', 65432))  # Get port from environment variable or use default

# def receive_data():
#     logging.info(f"Receiver starting up on {HOST}:{PORT}")
    
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         logging.info(f"Receiver listening on {HOST}:{PORT}")
        
#         while True:
#             conn, addr = s.accept()
#             logging.info(f"New connection from {addr}")
            
#             received_data = []
#             with conn:
#                 while True:
#                     data = conn.recv(1024)
#                     if not data:
#                         break
#                     received_values = data.decode().strip().split('\n')
#                     received_data.extend(received_values)
#                     for value in received_values:
#                         logging.info(f"Received value: {value}")
            
#             logging.info(f"Connection from {addr} closed.")
#             logging.info(f"Total received data points: {len(received_data)}")
#             logging.info(f"Received data: {received_data}")

# if __name__ == "__main__":
#     try:
#         receive_data()
#     except Exception as e:
#         logging.error(f"An error occurred: {e}", exc_info=True)












# # Network setup
# HOST = '0.0.0.0'  # Listen on all available interfaces
# PORT = int(os.environ.get('RECEIVER_PORT', 65432))  # Get port from environment variable or use default

# def receive_data():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         print(f"Receiver listening on {HOST}:{PORT}")
        
#         while True:
#             conn, addr = s.accept()
#             print(f"Connected by {addr}")
            
#             received_data = []
#             with conn:
#                 while True:
#                     data = conn.recv(1024)
#                     if not data:
#                         break
#                     received_values = data.decode().strip().split('\n')
#                     received_data.extend(received_values)
#                     print(f"Received: {received_values}")
            
#             print(f"Connection closed. Total received data: {received_data}")
#             print(f"Number of data points received: {len(received_data)}")

# if __name__ == "__main__":
#     receive_data()
