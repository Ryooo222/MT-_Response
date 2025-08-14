import matplotlib.pyplot as plt


depths = [0, 270, 1200, 12300, 14500]  # Depth of top surface [m] (the deepest layer at the end)
resistivities = [150, 9, 300, 5, 30]  # Resistivity of each layer [Ω·m]


# generate data
layer_depths_plot = []
layer_res_plot = []
for i in range(len(resistivities)):
    # top of layers
    layer_depths_plot.append(depths[i])
    layer_res_plot.append(resistivities[i])
    # botom of layers
    if i < len(resistivities) - 1:
        layer_depths_plot.append(depths[i+1])
        layer_res_plot.append(resistivities[i])

# draw the lowest layer as 2000 m wide
layer_depths_plot.append(depths[-1] + 2000)
layer_res_plot.append(resistivities[-1])


# plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.step(layer_res_plot, layer_depths_plot, where='post', color='k', linewidth=2)
ax.set_xscale('log')
ax.invert_yaxis()
ax.set_xlabel('Resistivity (Ωm)')
ax.set_ylabel('Depth (m)')
ax.set_xlim(1, 1e4)
ax.set_ylim(15500, 0)
ax.grid(True, which='both', ls='--', alpha=0.7)
plt.tight_layout()
plt.show()