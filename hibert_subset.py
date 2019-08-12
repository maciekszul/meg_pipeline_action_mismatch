import sys
import numpy as np
import mne
import os.path as op
from tools import files
from tqdm import tqdm

try:
    index = int(sys.argv[1])
except:
    print("incorrect file index")
    sys.exit()

path = "/cubric/scratch/c1557187/act_mis/MEG"
output_dir = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS"

subjects = files.get_folders_files(
    path,
    wp=False
)[0]
subjects.sort()

epoch_list_path = []
for subject in subjects:
    filep = op.join(
        path,
        subject,
        "new_v1"
    )
    epochs_list = files.get_files(
        filep,
        "epochs",
        "-epo.fif"
    )[2]
    epoch_list_path.extend(epochs_list)

epoch_list_path = [i for i in epoch_list_path if "-TF-" not in i]
epoch_list_path = [i for i in epoch_list_path if "-TD-" not in i]

info = mne.io.read_info(epoch_list_path[0])
info = mne.pick_info(info, mne.pick_channels(info["ch_names"], include=info["ch_names"][29:-3]))

sensor_groupings = {
    "A": [i for i in info["ch_names"] if "MLO" in i],
    "B": [i for i in info["ch_names"] if "MRO" in i],
    "C": [i for i in info["ch_names"] if "MLC" in i],
    "D": [i for i in info["ch_names"] if "MRC" in i],
    "E": [i for i in info["ch_names"] if "O" in i],
    "F": [i for i in info["ch_names"] if "C" in i],
    "G": info["ch_names"]
}

subsets = [
    "alpha",
    "beta",
    "theta",
    "low_gamma",
    "stimulus",
    "resp-alpha",
    "resp-beta",
    "resp-theta",
    "resp-low_gamma",
    "resp-stimulus",
]

# subsets = [
#     "beta",
#     "resp-beta",
# ]

for sub in subsets:
    print(sub)
    if "resp" not in sub:
        epoch_subset = [i for i in epoch_list_path if sub in i]
        epoch_subset = [i for i in epoch_subset if "resp" not in i]
        epoch_subset.sort()
    else:
        epoch_subset = [i for i in epoch_list_path if sub in i]
        epoch_subset.sort()
    regular_data = {key: [] for key in sensor_groupings.keys()}
    odd_data = {key: [] for key in sensor_groupings.keys()}

    for read_file in tqdm(epoch_subset):
        epochs = mne.read_epochs(
            read_file,
            preload=True
        )
        epochs = epochs.pick_types(
            ref_meg=False,
            stim=False,
            eog=False
        )
        epochs.apply_baseline((-0.5, 0.0))

        regular = epochs["30"].average()
        odd = epochs["40"].average()
        del epochs

        for key in sensor_groupings.keys():
            grp = sensor_groupings[key]
            r = np.sum(regular.copy().pick_channels(grp).data ** 2, axis=0)
            o = np.sum(odd.copy().pick_channels(grp).data ** 2, axis=0)
            regular_data[key].append(r)
            odd_data[key].append(o)
    regular_path = op.join(
        output_dir,
        "{}_regular.npy".format(sub)
    )

    odd_path = op.join(
        output_dir,
        "{}_odd.npy".format(sub)
    )

    np.save(regular_path, regular_data)
    np.save(odd_path, odd_data)