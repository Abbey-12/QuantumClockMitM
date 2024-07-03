import multiprocessing
import numpy as np
import time
import random
import matplotlib.pyplot as plt

def generate_correlated_poisson_signals(rate, duration, time_step, time_lag, correlation_strength):
    num_steps = int(duration / time_step)
    time_array = np.arange(0, duration, time_step)
    
    lambda_step = rate * time_step
    signal1 = np.random.poisson(lambda_step, num_steps)
    
    lag_steps = int(time_lag / time_step)
    signal2 = np.roll(signal1, lag_steps)
    
    noise = np.random.poisson(lambda_step * (1 - correlation_strength), num_steps)
    signal2 = np.random.poisson(signal2 * correlation_strength + noise)
    
    return time_array, signal1, signal2

def node_process(node_id, detection_times, comm_queue, clock_offset):
    local_clock = time.time() + clock_offset
    
    for t in detection_times:
        time.sleep(t)
        timestamp = time.time()
        comm_queue.put((node_id, timestamp))
    
    comm_queue.put((node_id, None))  # Signal end of detections

def classical_channel(comm_queue, node1_queue, node2_queue):
    while True:
        message = comm_queue.get()
        if message[1] is None:
            node1_queue.put(message)
            node2_queue.put(message)
            if message[0] == 1:
                break
        else:
            if message[0] == 0:
                node2_queue.put(message)
            else:
                node1_queue.put(message)

def clock_sync_node(node_id, detection_times, node_queue, comm_queue, initial_offset, result_queue):
    local_clock = time.time() + initial_offset
    time_differences = []
    timestamps = []

    node_process_instance = multiprocessing.Process(target=node_process, 
                                           args=(node_id, detection_times, comm_queue, initial_offset))
    node_process_instance.start()

    while True:
        message = node_queue.get()
        if message[1] is None:
            break
        
        other_node, other_timestamp = message
        local_timestamp = time.time()
        time_difference = local_timestamp - other_timestamp
        time_differences.append(time_difference)
        timestamps.append(local_timestamp)
    
    node_process_instance.join()
    
    avg_time_difference = np.mean(time_differences)
    print(f"Node {node_id} - Average time difference: {avg_time_difference:.6f} seconds")
    print(f"Node {node_id} - Estimated clock offset: {avg_time_difference/2:.6f} seconds")

    result_queue.put((timestamps, time_differences))

if __name__ == "__main__":
    # Simulation parameters
    rate = 10  # Average event rate (events per second)
    duration = 10  # Duration of the simulation (seconds)
    time_step = 0.01  # Time step (seconds)
    time_lag = 0.05  # Time lag between signals (seconds)
    correlation_strength = 0.8  # Strength of correlation (0 to 1)

    # Generate correlated Poisson signals
    _, signal1, signal2 = generate_correlated_poisson_signals(rate, duration, time_step, time_lag, correlation_strength)

    # Create communication queues
    comm_queue = multiprocessing.Queue()
    node1_queue = multiprocessing.Queue()
    node2_queue = multiprocessing.Queue()
    result_queue1 = multiprocessing.Queue()
    result_queue2 = multiprocessing.Queue()

    # Set initial clock offsets
    node1_offset = random.uniform(-0.1, 0.1)
    node2_offset = random.uniform(-0.1, 0.1)

    # Start the classical channel process
    channel_process = multiprocessing.Process(target=classical_channel, args=(comm_queue, node1_queue, node2_queue))
    channel_process.start()

    # Start the node processes
    node1_process = multiprocessing.Process(target=clock_sync_node, 
                                            args=(0, signal1, node1_queue, comm_queue, node1_offset, result_queue1))
    node2_process = multiprocessing.Process(target=clock_sync_node, 
                                            args=(1, signal2, node2_queue, comm_queue, node2_offset, result_queue2))

    node1_process.start()
    node2_process.start()

    # Wait for processes to finish
    node1_process.join()
    node2_process.join()
    channel_process.join()

    # Collect results
    timestamps1, time_differences1 = result_queue1.get()
    timestamps2, time_differences2 = result_queue2.get()

    # Calculate actual time offsets
    actual_offsets = [node2_offset - node1_offset] * len(timestamps1)

    # Calculate estimated time offsets
    estimated_offsets = [(td1 - td2) / 2 for td1, td2 in zip(time_differences1, time_differences2)]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps1, actual_offsets, label='Actual Offset', linestyle='--')
    plt.plot(timestamps1, estimated_offsets, label='Estimated Offset')
    plt.xlabel('Time (s)')
    plt.ylabel('Time Offset (s)')
    plt.title('Clock Synchronization: Actual vs Estimated Time Offset')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Actual clock offset: {node2_offset - node1_offset:.6f} seconds")
    print(f"Final estimated clock offset: {estimated_offsets[-1]:.6f} seconds")