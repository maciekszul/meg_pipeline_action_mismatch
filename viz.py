import mne
import numpy as np

# plotting topomaps

A = [i for i in epochs.ch_names if "MLO" in i]
B = [i for i in epochs.ch_names if "MRO" in i]
OC = [i for i in epochs.ch_names if "MZO" in i]
C = [i for i in epochs.ch_names if "MLC" in i]
D = [i for i in epochs.ch_names if "MRC" in i]
PC = [i for i in epochs.ch_names if "MZC" in i]

dummy_data = np.zeros(274)
ch_selection = np.zeros(274, dtype=bool)
ch_selection[mne.pick_channels(epochs.ch_names, PC)] = True

mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='r',
     linewidth=0, markersize=4)

mne.viz.plot_topomap(
    np.zeros(274),
    epochs.info,
    cmap="Greys",
    vmin=0,
    vmax=0,
    mask=ch_selection,
    mask_params=mask_params
)