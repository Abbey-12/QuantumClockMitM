sudo nsenter -t 9432 -n tcpdump -i eth0 -w captured-slave.pcap

# # command to see [PID] of the container
# docker inspect -f '{{.State.Pid}}' my_container

# docker exec -it attacker sysctl net.ipv4.ip_forward
