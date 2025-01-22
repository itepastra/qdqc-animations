import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *

# Parameters
b = 0.1  # Width of the Lorentzian distribution
h = 0.5  # Driving strength along S_x
b0 = 0  # Strength of the quantizing field
t_max = 10000  # Maximum time for evolution
time_steps = 200000  # Number of time steps
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
psi0 = (basis(2, 0)).unit()  # Initial state: |+>
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
sx_inf = h / (h + b) * expect(sx, psi0)
sz_inf = b / (h + b) * expect(sz, psi0)
print(f"Expected ⟨Sx⟩ -> {sx_inf}, ⟨Sz⟩ -> {sz_inf}")

# Plot the results
plt.figure()
plt.plot(tlist, avg_observables[:, 0], label="⟨Sx⟩")
plt.plot(tlist, avg_observables[:, 1], label="⟨Sy⟩")
plt.plot(tlist, avg_observables[:, 2], label="⟨Sz⟩")
plt.axhline(y=sx_inf, color="r", linestyle="--", label=f"⟨Sx⟩ → {sx_inf:.2f}")
plt.axhline(y=sz_inf, color="g", linestyle="--", label=f"⟨Sz⟩ → {sz_inf:.2f}")
plt.xlabel("Time")
plt.ylabel("Expectation values")
plt.title("Ensemble-averaged dynamics with quasi-static noise")
plt.legend()
plt.savefig("ensemble_averaged_dynamics.png")
print("Saved ensemble-averaged dynamics plot as 'ensemble_averaged_dynamics.png'.")
plt.show()
