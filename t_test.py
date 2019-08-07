import json
import os.path as op
import numpy as np
import pandas as pd
# from mne.stats import *
from mne.baseline import rescale
from scipy.stats import trim_mean, sem, ttest_rel, ttest_1samp
from tools import files
import matplotlib.pylab as plt
import seaborn as sns
from operator import itemgetter

data_path = "/cubric/scratch/c1557187/act_mis/RESULTS/TD_SOURCE_SPACE_PROC"
label_json = op.join(data_path, "label_order.json")
labels_csv = "/cubric/data/c1557187/meg_pipeline_action_mismatch/tools/atlas_glasser_2016.csv"

which_freq = "meg"

freq_dict = {
    "alpha": (7, 14),
    "beta": (14, 30),
    "delta": (None, 4),
    "theta": (4, 7),
    "low_gamma": (30, 80)
}

with open(label_json) as json_file:
    labels = json.load(json_file)

labels = eval(labels)

label_data = pd.read_csv(labels_csv)

mean_files = files.get_files(
    data_path,
    "means-{}".format(which_freq),
    ".npy"
)[2]
mean_files.sort()

odd, reg = [np.load(i).item() for i in mean_files]

odd = np.array([odd[key] for key in odd.keys()])
reg = np.array([reg[key] for key in reg.keys()])


times = np.linspace(-0.5, 2.6, num=776)

# reg = rescale(reg, times, (-0.1, 0.0))
# odd = rescale(odd, times, (-0.1, 0.0))

odd_mean = np.mean(odd, axis=0)
odd_sem = sem(odd, axis=0)

reg_mean = np.mean(reg, axis=0)
reg_sem = sem(reg, axis=0)




tick_pos = [125, 500, 525]

label_p = []
label_t = []
for ix, label_name in enumerate(labels):
    time_p = []
    time_t = []
    for t in range(reg.shape[2]):
        X = reg[:, ix, t]
        # Y = odd[:, ix, t]
        t, p = ttest_1samp(X, 0)
        time_p.append(p)
        time_t.append(t)
    label_p.append(time_p)
    label_t.append(time_t)

label_p = np.array(label_p)
label_t = np.array(label_t)
p_thr = label_p < 0.05
p_t = np.copy(label_t)
p_t[np.invert(p_thr)] = 0

lab_ = label_data.sort_values("LABEL_NAME")[["LABEL_NAME","PRIMARY_SECTION"]]
lab_.reset_index(inplace=True, drop=True)

order_L = lab_.sort_values("PRIMARY_SECTION").index.values
order_R = lab_.sort_values("PRIMARY_SECTION").index.values + 180

primary_section = lab_.sort_values("PRIMARY_SECTION").PRIMARY_SECTION.values

y_s_ticks = np.concatenate([primary_section[::-1], primary_section])

fig = plt.figure(figsize=(10, 11))
ax = fig.add_subplot(111)
im = ax.imshow(np.array([p_t[i] for i in np.concatenate([order_L[::-1], order_R])]), origin="upper", cmap="RdBu", aspect="auto", vmax=10, vmin=-10)
ax.set_xticks(tick_pos)
ax.set_xticklabels(times[tick_pos])
ax.axhline(179, color="red", lw=1)
[ax.axvline(i, color="black", lw=0.5) for i in tick_pos]
plt.colorbar(im)

plt.show()