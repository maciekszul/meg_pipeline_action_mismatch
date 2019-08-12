import sys
import numpy as np
import mne
import os.path as op
from tools import files
from scipy.stats import trim_mean, sem
from mne.stats import permutation_cluster_test
from mne.baseline import rescale
import matplotlib.pylab as plt
from matplotlib import gridspec

output_dir = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS"
epo_info_path = "/cubric/scratch/c1557187/act_mis/MEG/0001/new_v1/epochs-TD-001-epo.fif"

info = mne.io.read_info(epo_info_path)
ch_subset = mne.pick_types(info, ref_meg=False, eog=False, stim=False)
info = mne.pick_info(info, ch_subset)

sensor_groupings = {
    "A": [i for i in info["ch_names"] if "MLO" in i],
    "B": [i for i in info["ch_names"] if "MRO" in i],
    "C": [i for i in info["ch_names"] if "MLC" in i],
    "D": [i for i in info["ch_names"] if "MRC" in i],
    "E": [i for i in info["ch_names"] if "O" in i],
    "F": [i for i in info["ch_names"] if "C" in i],
    "G": info["ch_names"]
}


freq_bands = {
    "alpha": (9, 14, "Alpha"),
    "beta": (14, 30, "Beta"),
    "theta": (4, 7, "Theta"),
    "low_gamma": (30, 80, "Low Gamma"),
    "stimulus": (7, 9, "Stimulus")
}

freq_order = ("low_gamma", "beta", "alpha", "stimulus", "theta")
group_order = ("A", "B", "E", "C", "D", "F", "G")

times = np.linspace(-0.5, 1, num=376)  # respo specific
signal = np.linspace(-4, 10, num=376)

ticks = [0.0, 0.5, 1.0]  # respo specific
tick_labels = [str(i) for i in ticks]  # respo specific

reg_colour = "#00A4CC"
odd_colour = "#F95700"
sign_colour = "#00ff00"
non_sign_colour = "#cccccc"

ncols = len(group_order)
nrows = len(freq_order) + 1

gs = gridspec.GridSpec(
    ncols=ncols,
    nrows=nrows,
    wspace=0.1,
    hspace=0.2,
    width_ratios=[2]*7,
    height_ratios=[0.5]*6
)

figure = plt.figure(figsize=(18, 21))
for row, freq_key in enumerate(freq_order):

    for column, group_key in enumerate(group_order):
        # read the data
        data_files = files.get_files(
            output_dir,
            "resp-{}".format(freq_key), # respo specific
            ".npy"
        )[2]
        data_files.sort()

        odd, regular = [np.load(i).item() for i in data_files]
        if row == 0:
            ax = figure.add_subplot(
                gs[0, column],
                label="grouping{}".format(str(column))
            )
            dummy_data = np.zeros(274)
            ch_selection = np.zeros(274, dtype=bool)
            ch_selection[mne.pick_channels(info["ch_names"], sensor_groupings[group_key])] = True

            mask_params = dict(
                marker='o',
                markerfacecolor='w',
                markeredgecolor='r',
                linewidth=0, 
                markersize=1
            )
            mne.viz.plot_topomap(
                np.zeros(274),
                info,
                cmap="Greys",
                vmin=0,
                vmax=0,
                mask=ch_selection,
                mask_params=mask_params,
                axes=ax,
                show=False
            )
        row += 1
        ax = figure.add_subplot(
            gs[row, column],
            label="freq{}{}".format(str(row),str(column))
        )

        # data processing
        reg_data = np.array(regular[group_key]) * 1e14
        odd_data = np.array(odd[group_key]) * 1e14
        reg_mean = np.average(reg_data, axis=0)
        reg_sem = sem(reg_data, axis=0)
        rot_odd_mean = np.average(odd_data, axis=0)
        rot_odd_sem = sem(odd_data, axis=0)

        # plot results

        ax.plot(times, reg_mean, linewidth=1, color="red")
        ax.axhline(0, lw=0.5, color="black")
        ax.axvline(0, linestyle="--", linewidth=0.5, color="black")
        

        # ticks and labels
        if column == 0:
            freq_label = "{}\n{} - {} Hz".format(
                freq_bands[freq_key][2],
                str(freq_bands[freq_key][0]),
                str(freq_bands[freq_key][1])
            )
            ax.set_ylabel(freq_label)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == nrows-1:
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
        if column == ncols-1:
            ax.yaxis.set_label_position("right")
            # ax.yaxis.tick_right()
            # ax.yaxis.set_ticks([0, 5, 10])
            # ax.yaxis.set_ticklabels(["0", "5", "10"])
            ax.set_ylabel("GFP")
        plt.xlim((-0.5, 1))

plt.show()