import os.path as op
import matplotlib.pylab as plt
import matplotlib.contour as ticker
from tools import files
import numpy as np
from scipy.stats import trim_mean, sem
from mne.stats import permutation_cluster_1samp_test

output_dir = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS/SVM"

img_save = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS/VIZ_HELP"

dataset = "clk_vs_anti_new_baseline_low_gamma"

all_files = files.get_files(
    output_dir,
    dataset,
    ".npy"
)[2]

all_files.sort()

# remove_subj = [7, 32, 34, 35, 38, 40, 41, 43]
# remove_subj = [str(i).zfill(4) for i in remove_subj]

# all_files = [i for i in all_files if i[-8:-4] not in remove_subj]

data = []
for file in all_files:
    pp = np.load(file)
    pp = np.mean(pp, axis=0)
    data.append(pp)
data = np.array(data)

times = np.linspace(-0.6, 2.6, num=801)

minimax = (0.25, 0.5, 0.75)

tg_mean = np.mean(data, axis=0)

print("start")

# perm t test
t_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
    data - 0.50,
    connectivity=None,
    step_down_p=0,
    tail=0, 
    n_jobs=-1,
    verbose=True
)

print("done")

threshold = 0.05
bool_map = np.zeros(tg_mean.shape)
cluster_array = np.array(clusters)
cluster_amount = len(np.where(cluster_p < threshold))
cluster_sig = cluster_array[np.where(cluster_p < threshold)]
cluster_mask = np.any(cluster_sig, axis=0)
bool_map[cluster_mask] = 0.1

fig, ax = plt.subplots(figsize=(10,10))

terrain = ax.imshow(
    tg_mean, 
    interpolation="lanczos", 
    origin="lower",
    cmap='RdBu_r',
    extent=times[[0, -1, 0, -1]],
    vmin=minimax[0], 
    vmax=minimax[2]
)

contours = ax.contour(
    bool_map,
    1,
    linewidths=0.2,
    colors="black",
    linestyles="solid",
    origin="lower",
    extent=times[[0, -1, 0, -1]]
)


ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal Generalization')

ax.axvline(0, linewidth=.5, linestyle='--', color='black')
ax.axhline(0, linewidth=.5, linestyle='--', color='black')

ax.axvline(1.5, linewidth=.5, linestyle='--', color='black')
ax.axhline(1.5, linewidth=.5, linestyle='--', color='black')

ax.axvline(1.6, linewidth=.5, linestyle='--', color='black')
ax.axhline(1.6, linewidth=.5, linestyle='--', color='black')

colorbar = plt.colorbar(terrain, ax=ax, ticks=minimax, shrink=0.75)
colorbar.set_label("Classification performance [Accuracy]")

filenameTG = op.join(
    img_save,
    "TG_{}.svg".format(dataset)
)

plt.show()
plt.savefig(filenameTG, bbox_inches="tight")

diag_sem = sem(np.diagonal(data), axis=1)
diag_mean = np.diagonal(tg_mean)


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(times, diag_mean)
ax.fill_between(
    times,
    diag_mean-diag_sem,
    diag_mean+diag_sem,
    alpha=0.2,
    linewidth=0
)
ax.axhline(0.5, linewidth=.5, linestyle='-', color='black')

ax.axvline(0, linewidth=.5, linestyle='--', color='black')
ax.axvline(1.5, linewidth=.5, linestyle='--', color='black')
ax.axvline(1.6, linewidth=.5, linestyle='--', color='black')

ax.set_ylabel("Classification performance [Accuracy]")
ax.set_xlabel("Training Time (s)")

plt.ylim([0.45, 0.75])
plt.xlim([-0.5, 2.6])

filenameCLAS = op.join(
    img_save,
    "CLAS_{}.svg".format(dataset)
)

plt.show()
plt.savefig(filenameCLAS, bbox_inches="tight")