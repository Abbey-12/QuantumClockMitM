#!/bin/bash

# Wait for the IP address of the other container
while [ -z "$OTHER_CONTAINER_IP" ]; do
    sleep 1
    OTHER_CONTAINER_IP=$(getent hosts other_container | awk '{ print $1 }')
done

# Send clock information
while true; do
    echo "Clock from $(hostname): $(date)" | nc -u -b $OTHER_CONTAINER_IP 12345
    sleep 1
done