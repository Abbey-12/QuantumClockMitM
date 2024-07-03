import time
import socket
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

MITM_HOST = '0.0.0.0'
MITM_PORT = 65431
RECEIVER_HOST = 'reciver'  # Use the actual receiver hostname or IP
RECEIVER_PORT = 65432

def modify_data(data):
    """Modify some of the received data."""
    values = data.decode().strip().split('\n')
    modified_values = []
    for value in values:
        if random.random() < 0.2:  # 20% chance to modify a value
            modified_value = str(int(value) + random.randint(-2, 2))
            logging.info(f"Modified value: {value} -> {modified_value}")
            modified_values.append(modified_value)
        else:
            modified_values.append(value)
    return '\n'.join(modified_values).encode() + b'\n'

   
def eavesdrop():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', MITM_PORT))
        s.listen()
        print(f"MITM listening on port {MITM_PORT}")
        
        while True:
            conn, addr = s.accept()
            print(f"Connection from {addr}")
            
            with conn:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver_socket:
                    receiver_socket.connect((RECEIVER_HOST, RECEIVER_PORT))
                    
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        
                        modified_values= modify_data(data)
                        
                        # Forward to receiver
                        for value in modified_values:
                            receiver_socket.sendall(f"{value}\n".encode())
                            time.sleep(0.1)

if __name__ == "__main__":
    eavesdrop()
