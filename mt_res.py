import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

obs_df = pd.read_excel("P310.xlsx")
obs_df.drop(obs_df.index[0], inplace=True)

def compute_mt_response(frequencies, depths, resistivities):
    mu0 = 4 * np.pi * 1e-7  # Permeability of Vacuum (H/m)
    angl_freq = 2 * np.pi * frequencies
    n_layers = len(resistivities)

    # Bottom layer impedance
    Z = np.sqrt(1j * angl_freq * mu0 * resistivities[-1])

    # Calculate impedance recursively from top to bottom
    for i in reversed(range(n_layers - 1)):
        h = depths[i+1] - depths[i]
        r = resistivities[i]
        k = np.sqrt(1j * angl_freq * mu0 / r)
        w = k * h
        Z_next = np.sqrt(1j * angl_freq * mu0 * r)
        Z = Z_next * (Z + Z_next * np.tanh(w)) / (Z_next + Z * np.tanh(w))

    # Calculate apparent resistivity and phase
    rho_a = (np.abs(Z)**2) / (mu0 * angl_freq)
    phase = np.angle(Z, deg=True)

    return rho_a, phase

# ----------------------------
# Input: Depth of top surface of layer [m] and resistivity [Ω·m]

depths = [0, 270, 1800, 12300, 14500]  # Depth of top surface [m] (the deepest layer at the end)
resistivities = [150, 9, 300, 5, 30]  # Resistivity of each layer [Ω·m]

# nice numbers
# depths = [0, 270, 1200, 12300, 14500]
# resistivities = [150, 9, 300, 5, 30]


# Frequency range (Hz)
frequencies = np.logspace(-5, 5, 200)

# calculate MT response
rho_a, phase = compute_mt_response(frequencies, depths, resistivities)

# ----------------------------
# log scale RMS
def log_rms(y_model, y_obs):
    y_model = np.asarray(y_model)
    y_obs = np.asarray(y_obs)
    if y_model.shape != y_obs.shape:
        # xticks
        x_model = frequencies if 'frequencies' in globals() else np.arange(len(y_model))
        x_obs = obs_df["f(hz)"].values if len(y_obs) == len(obs_df["f(hz)"].values) else np.arange(len(y_obs))
        x_obs = np.asarray(x_obs).astype(float)
        y_obs = np.asarray(y_obs).astype(float)
        y_obs = np.interp(x_model, x_obs, y_obs)
    # drop <= 0
    mask = (y_model > 0) & (y_obs > 0)
    return np.sqrt(np.mean((np.log10(y_model[mask]) - np.log10(y_obs[mask]))**2))

rms_rho = log_rms(rho_a, obs_df["Rhoxy ohm-m."].values)
rms_phase = log_rms(phase, obs_df["PHASExy Deg."].values)
print(f"log RMS (apparent resistivity): {rms_rho:.4f}")
print(f"log RMS (phase): {rms_phase:.4f}")


# plot -----------------------

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 6.5))

# Apparent Resistivity
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Apparent Resistivity [Ω·m]', color='blue')
ax1.plot(frequencies, rho_a, label="model", c="b")
ax1.plot(obs_df["f(hz)"], obs_df["Rhoxy ohm-m."], label="obs", c="b", linestyle='--')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title(f'Apparent Resistivity)')
ax1.set_ylim(1e0, 1e3)
ax1.grid(True, which='both', ls='--', alpha=0.5)
ax1.legend()

# Phase
ax2.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase [deg]', color='red')
ax2.plot(frequencies, phase, label='model', c='r')
ax2.plot(obs_df["f(hz)"], obs_df["PHASExy Deg."], label="obs", c="red", linestyle='--')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_title(f'Phase')
ax2.set_ylim(0, 90)
ax2.grid(True, which='both', ls='--', alpha=0.5)
ax2.legend()

# 45 degree line
ax2.axhline(45, color='gray', linewidth=1, linestyle='--', alpha=0.8)

plt.suptitle('MT Response (1D Layered Earth)\n top of 3rd layer: 1800 m')
plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()
plt.savefig("mt_res.png", dpi=300)