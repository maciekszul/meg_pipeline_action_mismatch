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
from tqdm import tqdm

try:
    key_index = int(sys.argv[1])
except:
    print("incorrect file index")
    sys.exit()

output_dir = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS"
img_save = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS/VIZ_HELP"
epo_info_path = "/cubric/scratch/c1557187/act_mis/MEG/0001/new_v1/epochs-TD-001-epo.fif"

data_files = files.get_files(
    output_dir,
    "TD",
    ".npy"
)[2]
data_files.sort()

odd, regular = [np.load(i).item() for i in data_files]

keys = list(regular.keys())


key = keys[key_index]

print(key)

info = mne.io.read_info(epo_info_path)
ch_subset = mne.pick_types(info, ref_meg=False, eog=False, stim=False)
info = mne.pick_info(info, ch_subset)

sensor_groupings = {
    "A": [i for i in info["ch_names"] if "MLO" in i],
    "B": [i for i in info["ch_names"] if "MRO" in i],
    "OC": [i for i in info["ch_names"] if "MZO" in i],
    "C": [i for i in info["ch_names"] if "MLC" in i],
    "D": [i for i in info["ch_names"] if "MRC" in i],
    "PC": [i for i in info["ch_names"] if "MZC" in i]
}

dummy_data = np.zeros(274)
ch_selection = np.zeros(274, dtype=bool)
ch_selection[mne.pick_channels(info["ch_names"], sensor_groupings[key])] = True

mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='r',
     linewidth=0, markersize=4)

reg_colour = "#00A4CC"
odd_colour = "#F95700"
sign_colour = "#00ff00"
non_sign_colour = "#cccccc"

reg_data = np.array(regular[key]) * 1e14
odd_data = np.array(odd[key]) * 1e14

rot_range = (0, 525)
obs_range = (525, 801)
rot_times = np.linspace(-0.6, 1.5, num=np.diff(rot_range)[0])
obs_times = np.linspace(-0.1, 1.0, num=np.diff(obs_range)[0])

rot_reg = rescale(reg_data[:,rot_range[0]:rot_range[1]], rot_times, (-0.6, -0.5), mode="mean")
rot_reg_mean = np.average(rot_reg, axis=0)
rot_reg_sem = sem(rot_reg, axis=0)
obs_reg = rescale(reg_data[:,obs_range[0]:obs_range[1]], obs_times, (-0.1, 0.0), mode="mean")
obs_reg_mean = np.average(obs_reg, axis=0)
obs_reg_sem = sem(obs_reg, axis=0)
rot_odd = rescale(odd_data[:,rot_range[0]:rot_range[1]], rot_times, (-0.6, -0.5), mode="mean")
rot_odd_mean = np.average(rot_odd, axis=0)
rot_odd_sem = sem(rot_odd, axis=0)
obs_odd = rescale(odd_data[:,obs_range[0]:obs_range[1]], obs_times, (-0.1, 0.0), mode="mean")
obs_odd_mean = np.average(obs_odd, axis=0)
obs_odd_sem = sem(obs_odd, axis=0)

threshold=2.0

rot_T_obs, rot_clusters, rot_cluster_p_values, rot_H0 = permutation_cluster_test(
    [rot_reg, rot_odd], 
    n_permutations=5000, 
    threshold=threshold, 
    tail=0, 
    n_jobs=-1
)

obs_T_obs, obs_clusters, obs_cluster_p_values, obs_H0 = permutation_cluster_test(
    [obs_reg, obs_odd], 
    n_permutations=5000, 
    threshold=threshold, 
    tail=0, 
    n_jobs=-1
)



gs = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.1, width_ratios=[0.4, 0.2, 0.4])
figure = plt.figure(figsize=(15, 5))

ax_rot = figure.add_subplot(gs[0])

for i_c, c in enumerate(rot_clusters):
    c = c[0]
    if rot_cluster_p_values[i_c] < 0.05:
        ax_rot.axvspan(
            rot_times[c.start], 
            rot_times[c.stop-1],
            color=sign_colour,
            alpha=0.2
        )
    else:
        ax_rot.axvspan(
            rot_times[c.start], 
            rot_times[c.stop-1],
            color=non_sign_colour,
            alpha=0.2
        )

ax_rot.plot(rot_times, rot_reg_mean, linewidth=1.5, color=reg_colour, label="Regular")
ax_rot.fill_between(rot_times, rot_reg_mean-rot_reg_sem, rot_reg_mean+rot_reg_sem, color=reg_colour, alpha=0.2, linewidth=0)
ax_rot.plot(rot_times, rot_odd_mean, linewidth=1.5, color=odd_colour, label="Odd")
ax_rot.fill_between(rot_times, rot_odd_mean-rot_odd_sem, rot_odd_mean+rot_odd_sem, color=odd_colour, alpha=0.2, linewidth=0)
ax_rot.axhline(0, linewidth=0.5, color="black")
ax_rot.axvline(0, linestyle="--", linewidth=0.5, color="black")
plt.title("Movement phase")
plt.legend(loc=1)
plt.xlabel("Time[s]")
plt.ylabel("fT")
plt.ylim(-10, 10)
plt.xlim(-0.6, 1.5)

ax_obs = figure.add_subplot(gs[2])

for i_c, c in enumerate(obs_clusters):
    c = c[0]
    if obs_cluster_p_values[i_c] < 0.05:
        ax_obs.axvspan(
            obs_times[c.start], 
            obs_times[c.stop-1],
            color=sign_colour,
            alpha=0.2
        )
    else:
        ax_obs.axvspan(
            obs_times[c.start], 
            obs_times[c.stop-1],
            color=non_sign_colour,
            alpha=0.2
        )

ax_obs.plot(obs_times, obs_reg_mean, linewidth=1.5, color=reg_colour, label="Regular")
ax_obs.fill_between(obs_times, obs_reg_mean-obs_reg_sem, obs_reg_mean+obs_reg_sem, color=reg_colour, alpha=0.2, linewidth=0)
ax_obs.plot(obs_times, obs_odd_mean, linewidth=1.5, color=odd_colour, label="Odd")
ax_obs.fill_between(obs_times, obs_odd_mean-obs_odd_sem, obs_odd_mean+obs_odd_sem, color=odd_colour, alpha=0.2, linewidth=0)
ax_obs.axhline(0, linewidth=0.5, color="black")
ax_obs.axvline(0, linestyle="--", linewidth=0.5, color="black")
plt.title("Observation phase")
plt.legend(loc=1)
plt.xlabel("Time[s]")
plt.ylabel("fT")
plt.ylim(-10, 10)
plt.xlim(-0.1, 1.0)

ax_subset = figure.add_subplot(gs[1])

mne.viz.plot_topomap(
    np.zeros(274),
    info,
    cmap="Greys",
    vmin=0,
    vmax=0,
    mask=ch_selection,
    mask_params=mask_params,
    axes=ax_subset,
    show=False
)
plt.title("Channel subset")

plt.tight_layout(w_pad=0.2, h_pad=0.2)

out_path = op.join(
    img_save,
    "ERF_ttest_{}.svg".format(key)
)
plt.savefig(out_path, bbox_inches="tight")
plt.show()