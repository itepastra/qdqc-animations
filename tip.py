#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *

# Parameters
b = 0.001  # Width of the Lorentzian distribution
h = 0.5  # Driving strength along S_x
b0 = 5  # Strength of the quantizing field
t_max = 1000  # Maximum time for evolution
time_steps = 20000  # Number of time steps
tlist = np.linspace(0, t_max, time_steps)  # Time array
ensemble_size = 1000  # Number of ensemble members


# Define Lorentzian noise sampler
def lorentzian_noise(b, size=1):
    print(f"Generating {size} Lorentzian noise samples with width {b}")
    noise = scipy.stats.cauchy.rvs(loc=0, scale=b, size=size)
    print(f"Generated noise samples: {noise[:5]}... (showing first 5)")
    return noise


# System setup
sx, sy, sz = sigmax(), sigmay(), sigmaz()
psi0 = (basis(2, 0) - basis(2, 1)).unit()  # Initial state: |0>
print(f"Initial state: {psi0}")

# Hamiltonian components
H0 = b0 * sz  # Base Hamiltonian (detuning)
Hd = h * sx  # Driving term along S_x
print(f"Base Hamiltonian H0: {H0}")
print(f"Driving Hamiltonian Hd: {Hd}")

# Generate static noise for the ensemble
betas = lorentzian_noise(b, ensemble_size)
print(f"Generated static noise samples for ensemble: {betas[:5]}... (showing first 5)")

# Solve the dynamics for each ensemble member
observables = np.zeros(
    (ensemble_size, time_steps, 3)
)  # Store expectation values for sx, sy, sz

for i, beta in enumerate(betas):
    print(f"Simulating for ensemble member {i + 1}/{ensemble_size} with beta={beta}")
    H = H0 + Hd + beta * sz  # Static noise term included directly in the Hamiltonian
    result = mesolve(H, psi0, tlist, [], [])

    # Store observables
    for t, state in enumerate(result.states):
        observables[i, t, 0] = expect(sx, state)
        observables[i, t, 1] = expect(sy, state)
        observables[i, t, 2] = expect(sz, state)

print(observables)
# Average over the ensemble
avg_observables = np.mean(observables, axis=0)
stdev_observables = np.std(observables, axis=0)

# Calculate pointer state expectations
nx0 = h / np.sqrt(b0 * b0 + h * h)
nz0 = b0 / np.sqrt(b0 * b0 + h * h)
sn0 = sx * nx0 + sz * nz0
sx_inf = nx0 * expect(sn0, psi0)
sz_inf = nz0 * expect(sn0, psi0)
print(f"Expected ⟨Sx⟩ -> {sx_inf}, ⟨Sz⟩ -> {sz_inf}")

# Plot the results
plt.figure()
plt.plot(tlist, avg_observables[:, 0], color="r", label="⟨Sx⟩")
# plt.plot(tlist, avg_observables[:, 1], label="⟨Sy⟩")
plt.plot(tlist, avg_observables[:, 2], color="g", label="⟨Sz⟩")
plt.axhline(y=sx_inf, color="r", linestyle="--", label=f"⟨Sx⟩ → {sx_inf:.2f}")
plt.axhline(y=sz_inf, color="g", linestyle="--", label=f"⟨Sz⟩ → {sz_inf:.2f}")
plt.xlabel("Time")
plt.ylabel("Expectation values")
plt.title("Ensemble-averaged dynamics with quasi-static noise")
plt.legend()
# Add text annotations for parameters and initial state
textstr = f"Parameters:\nb = {b}\nh = {h}\nInitial state: {psi0.full().flatten()}"
plt.gcf().text(0.15, 0.5, textstr, fontsize=10, bbox=dict(facecolor="white", alpha=0.7))
plt.savefig("ensemble_averaged_resonant_strong_minus.png")
print("Saved ensemble-averaged dynamics plot as 'ensemble_averaged_dynamics.png'.")
plt.show()
