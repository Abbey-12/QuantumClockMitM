import numpy as np
from qutip import *
from qutip.channels import amplitude_damping_channel

# Define initial Bell state (|00⟩ + |11⟩) / sqrt(2)
bell_state = (basis(2, 0) * basis(2, 0).dag() + basis(2, 1) * basis(2, 1).dag()) / np.sqrt(2)

print(bell_state)

# Simulate a noisy quantum channel (example with amplitude damping)
noise_parameter = 0.1
# Create the amplitude damping channel
ad_channel = amplitude_damping_channel(1 - noise_parameter)
# Apply the channel to one qubit of the Bell state
channel = tensor(qeye(2), to_super(ad_channel))

# Transmit the Bell state through the channel
transmitted_state = channel * bell_state.ptrace([0, 1])

# Eve intercepts and measures one photon (example measurement on Alice's photon)
eve_measurement = tensor(sigmax(), qeye(2))
collapsed_state = eve_measurement * transmitted_state

# Eve prepares a fake entangled state (Bell state |01⟩)
fake_bell_state = (basis(2, 0) * basis(2, 1).dag() + basis(2, 1) * basis(2, 0).dag()) / np.sqrt(2)

# Perform entanglement swapping (Bell state measurement between intercepted and fake photons)
entanglement_swap = bell_state * fake_bell_state.dag()
final_state = entanglement_swap * collapsed_state

# Bob and Charlie measure their photons (example measurement on Bob's photon)
measurement_bob = tensor(sigmax(), qeye(2))
measurement_charlie = tensor(qeye(2), sigmax())
bob_result = measurement_bob * final_state
charlie_result = measurement_charlie * final_state

# Calculate probabilities or perform further analysis
prob_bob = abs(bob_result.norm()) ** 2
prob_charlie = abs(charlie_result.norm()) ** 2

print(f"Probability of Bob's measurement result: {prob_bob}")
print(f"Probability of Charlie's measurement result: {prob_charlie}")