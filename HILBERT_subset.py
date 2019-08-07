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
        "epochs-TF",
        "-epo.fif"
    )[2]
    epoch_list_path.extend(epochs_list)

freq_dict = {
    "alpha": (7, 14),
    "beta": (14, 30),
    "delta": (None, 4),
    "theta": (4, 7),
    "low_gamma": (30, 80),
    "stimulus_induced": (7, 9)
}


info = mne.io.read_info(epoch_list_path[0])

sensor_groupings = {
    "A": [i for i in info["ch_names"] if "MLO" in i],
    "B": [i for i in info["ch_names"] if "MRO" in i],
    "OC": [i for i in info["ch_names"] if "MZO" in i],
    "C": [i for i in info["ch_names"] if "MLC" in i],
    "D": [i for i in info["ch_names"] if "MRC" in i],
    "PC": [i for i in info["ch_names"] if "MZC" in i]
}

for which_freq in freq_dict.keys():
    f_low, f_high = freq_dict[which_freq]
    regular_data = {key: [] for key in sensor_groupings.keys()}
    odd_data = {key: [] for key in sensor_groupings.keys()}

    for read_file in tqdm(epoch_list_path):
        epochs = mne.read_epochs(
            read_file,
            preload=True
        )
        epochs = epochs.pick_types(ref_meg=False)

        epochs = epochs.filter(
            f_low,
            None,
            method="fir",
            phase="minimum",
            n_jobs=-1
        )

        epochs = epochs.filter(
            None,
            f_high,
            method="fir",
            phase="minimum",
            n_jobs=-1
        )

        regular = epochs["30"].average()
        odd = epochs["40"].average()

        for key in sensor_groupings.keys():
            grp = sensor_groupings[key]
            r = np.average(regular.copy().pick_channels(grp).data, axis=0)
            o = np.average(odd.copy().pick_channels(grp).data, axis=0)

            regular_data[key].append(r)
            odd_data[key].append(o)


    regular_path = op.join(
        output_dir,
        "{}_regular.npy".format(which_freq)
    )
    odd_path = op.join(
        output_dir,
        "{}_odd.npy".format(which_freq)
    )

    np.save(regular_path, regular_data)
    np.save(odd_path, odd_data)