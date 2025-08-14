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
# depths = [0, 250, 1200, 10000]  # Depth of top surface [m] (the deepest layer at the end)
# resistivities = [250, 10, 1000, 20]  # Resistivity of each layer [Ω·m]

depths = [0, 250, 1200, 9000, 13000]  # Depth of top surface [m] (the deepest layer at the end)
resistivities = [200, 10, 2000, 10, 40]  # Resistivity of each layer [Ω·m]


# Frequency range (Hz)
frequencies = np.logspace(-5, 5, 200)

# calculate MT response
rho_a, phase = compute_mt_response(frequencies, depths, resistivities)

# ----------------------------

def log_rms(y_model, y_obs):
    # 配列化
    y_model = np.asarray(y_model)
    y_obs = np.asarray(y_obs)
    # 長さが違う場合はy_modelのx軸（frequencies）に合わせて補間
    if y_model.shape != y_obs.shape:
        # x軸取得
        x_model = frequencies if 'frequencies' in globals() else np.arange(len(y_model))
        x_obs = obs_df["f(hz)"].values if len(y_obs) == len(obs_df["f(hz)"].values) else np.arange(len(y_obs))
        # 明示的にfloat型へ変換
        x_obs = np.asarray(x_obs).astype(float)
        y_obs = np.asarray(y_obs).astype(float)
        y_obs = np.interp(x_model, x_obs, y_obs)
    # ゼロや負値を除外（log10のため）
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
ax1.set_title(f'Apparent Resistivity (RMS: {rms_rho:.4f})')
ax1.set_ylim(1e0, 1e3)
ax1.grid(True, which='both', ls='--', alpha=0.5)

# Phase
ax2.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase [deg]', color='red')
ax2.plot(frequencies, phase, 'r')
ax2.plot(obs_df["f(hz)"], obs_df["PHASExy Deg."], label="obs", c="red", linestyle='--')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_title(f'Phase (RMS: {rms_phase:.4f})')
ax2.set_ylim(0, 90)
ax2.grid(True, which='both', ls='--', alpha=0.5)

# 45 degree line
ax2.axhline(45, color='gray', linewidth=1, linestyle='--', alpha=0.8)

plt.suptitle('MT Response (1D Layered Earth)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()
plt.savefig("mt_res.png", dpi=300)

#RMSなど残差を計算するにあたり、対数軸上で計算する方がよい！！

'''

# 層構造プロット用データ作成
layer_depths_plot = []
layer_res_plot = []
for i in range(len(resistivities)):
    # 各層の上面
    layer_depths_plot.append(depths[i])
    layer_res_plot.append(resistivities[i])
    # 各層の下面（最終層以外）
    if i < len(resistivities) - 1:
        layer_depths_plot.append(depths[i+1])
        layer_res_plot.append(resistivities[i])

# 最下層を少し延長
layer_depths_plot.append(depths[-1] + 50)
layer_res_plot.append(resistivities[-1])

# km単位に変換（正の値、0が地表）
layer_depths_plot_km = [d / 1000 for d in layer_depths_plot]

# 3つ並べてプロット
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2, 2]})

# 層構造プロット（左端）
ax0.step(layer_res_plot, layer_depths_plot_km, where='post', color='k')
ax0.set_xlabel('Resistivity [Ω·m]')
ax0.set_ylabel('Depth [km]')
ax0.set_xlim(1, 1e4)
ax0.set_ylim(1e-2, 1e3)  # 0km～1000km
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.invert_yaxis()  # 地表が上、深部が下
ax0.set_title('Layered Structure')
ax0.grid(True, which='both', ls='--', alpha=0.5)

# 見かけ比抵抗
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Apparent Resistivity [Ω·m]', color='blue')
ax1.plot(frequencies, rho_a, 'b')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Apparent Resistivity')
ax1.set_ylim(10, 1000)
ax1.grid(True, which='both', ls='--', alpha=0.5)

# 位相
ax2.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase [deg]', color='red')
ax2.plot(frequencies, phase, 'r')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_title('Phase')
ax2.set_ylim(0, 90)
ax2.grid(True, which='both', ls='--', alpha=0.5)
ax2.axhline(45, color='gray', linewidth=1, linestyle='--', alpha=0.8)

plt.suptitle('MT Response (1D Layered Earth)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
'''