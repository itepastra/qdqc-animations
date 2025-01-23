#!/usr/bin/env python
from datetime import datetime
import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *
import itertools
from multiprocessing import Pool

bs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
hs = [0.5]
b0s = [0.0, 0.05, 0.5, 5.0]
psi0s = [
    basis(2, 0),
    (basis(2, 0) + basis(2, 1)).unit(),
    (3 * basis(2, 0) + basis(2, 1)).unit(),
]
noises = ["lorentz", "gauss"]

# Parameters
parameter_sets = [
    # Add more parameter combinations here
    {"b": b, "h": h, "b0": b0, "psi0": psi0, "noise": noise}
    for (b, h, b0, psi0, noise) in list(itertools.product(bs, hs, b0s, psi0s, noises))
]

t_max = 10000  # Maximum time for evolution
time_steps = 20000  # Number of time steps
tlist = np.linspace(0, t_max, time_steps)  # Time array
ensemble_size = 1000  # Number of ensemble members


def lorentzian_noise(b, noise, size=1):
    nfunc = None
    if noise == "lorentz":
        nfunc = scipy.stats.cauchy
    if noise == "gauss":
        nfunc = scipy.stats.norm
    print(f"Generating {size} {noise} noise samples with width {b}")
    noise = nfunc.rvs(loc=0, scale=b, size=size)
    print(f"Generated noise samples: {noise[:5]}... (showing first 5)")
    return noise


timings = []


def simulate(parameters):
    b = parameters["b"]
    h = parameters["h"]
    b0 = parameters["b0"]
    psi0 = parameters["psi0"]
    noise = parameters["noise"]

    print(
        f"Starting simulation with parameters: b={b}, h={h}, b0={b0}, psi0={psi0.full().flatten()}"
    )

    # System setup
    sx, sy, sz = sigmax(), sigmay(), sigmaz()

    # Hamiltonian components
    H0 = b0 * sz  # Base Hamiltonian (detuning)
    Hd = h * sx  # Driving term along S_x

    # Generate static noise for the ensemble
    betas = lorentzian_noise(b, noise, ensemble_size)

    # Solve the dynamics for each ensemble member
    observables = np.zeros(
        (ensemble_size, time_steps, 3)
    )  # Store expectation values for sx, sy, sz

    for i, beta in enumerate(betas):
        start = datetime.now()
        if i % 25 == 0:
            print(
                f"Processing ensemble (b={b}, h={h}, B0={b0}) member {i + 1}/{ensemble_size} with beta={beta}"
            )
        H = (
            H0 + Hd + beta * sz
        )  # Static noise term included directly in the Hamiltonian
        result = mesolve(H, psi0, tlist, [], [])

        # Store observables
        for t, state in enumerate(result.states):
            observables[i, t, 0] = expect(sx, state)
            observables[i, t, 1] = expect(sy, state)
            observables[i, t, 2] = expect(sz, state)
        end = datetime.now()
        timings.append(
            {
                "b": b,
                "h": h,
                "b0": b0,
                "noise": noise,
                "beta": beta,
                "start": start,
                "end": end,
            }
        )

    print(f"Finished simulation for parameters: b={b}, h={h}, b0={b0}")

    # Average over the ensemble
    avg_observables = np.mean(observables, axis=0)
    stdev_observables = np.std(observables, axis=0)

    # Calculate pointer state expectations
    # nx0 = h / np.sqrt(b0 * b0 + h * h)
    # nz0 = b0 / np.sqrt(b0 * b0 + h * h)
    nx0 = h / (b + h)
    nz0 = b / (b + h)

    sn0 = sx * nx0 + sz * nz0
    sx_inf = nx0 * expect(sx, psi0)
    sz_inf = nz0 * expect(sz, psi0)

    # Plot the results
    plt.figure()
    plt.plot(tlist, avg_observables[:, 0], color="r", label="⟨Sx⟩")
    plt.plot(tlist, avg_observables[:, 2], color="g", label="⟨Sz⟩")
    plt.axhline(y=sx_inf, color="r", linestyle="--", label=f"⟨Sx⟩ → {sx_inf:.2f}")
    plt.axhline(y=sz_inf, color="g", linestyle="--", label=f"⟨Sz⟩ → {sz_inf:.2f}")
    plt.xlim(1, t_max)
    plt.xscale("symlog")
    plt.xlim(1, t_max)
    plt.xlabel("Time")
    plt.ylabel("Expectation values")
    plt.title(f"b={b}, h={h}, b0={b0}")
    # Add text annotations for parameters and initial state
    textstr = f"Parameters:\nb = {b}\nh = {h}\nb0 = {b0}\nInitial state: {psi0.full().flatten()}"
    plt.gcf().text(
        0.15, 0.5, textstr, fontsize=10, bbox=dict(facecolor="white", alpha=0.7)
    )
    plt.legend()
    filename = f"ensemble_{noise}_b_{b}_h_{h}_b0_{b0}_psi0_{psi0.full().flatten()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot as '{filename}'.")


# Use multiprocessing to run simulations in parallel
with Pool() as pool:
    pool.map(simulate, parameter_sets)

print(timings)
