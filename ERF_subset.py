import sys
import numpy as np
import mne
from mne.baseline import rescale
import os.path as op
from tools import files
from tqdm import tqdm

# try:
#     index = int(sys.argv[1])
# except:
#     print("incorrect file index")
#     sys.exit()

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
        "epochs-TD",
        "-epo.fif"
    )[2]
    epoch_list_path.extend(epochs_list)

info = mne.io.read_info(epoch_list_path[0])

sensor_groupings = {
    "A": [i for i in info["ch_names"] if "MLO" in i],
    "B": [i for i in info["ch_names"] if "MRO" in i],
    "OC": [i for i in info["ch_names"] if "MZO" in i],
    "C": [i for i in info["ch_names"] if "MLC" in i],
    "D": [i for i in info["ch_names"] if "MRC" in i],
    "PC": [i for i in info["ch_names"] if "MZC" in i]
}

regular_data = {key: [] for key in sensor_groupings.keys()}
odd_data = {key: [] for key in sensor_groupings.keys()}

for read_file in tqdm(epoch_list_path):
    epochs = mne.read_epochs(
        read_file,
        preload=True
    )

    epochs = epochs.pick_types(ref_meg=False)
    data = epochs.get_data()
    times = np.linspace(-0.6, 2.6, num=801)
    data = rescale(data, times, (-0.6, -0.5), mode="mean")

    data[:,:,525:] = rescale(data[:,:,525:], times[525:], (1.5, 1.6), mode="mean")

    epochs = mne.EpochsArray(
        data,
        epochs.info,
        events=epochs.events,
        tmin=epochs.tmin
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
    "TD_regular.npy"
)
odd_path = op.join(
    output_dir,
    "TD_odd.npy"
)

np.save(regular_path, regular_data)
np.save(odd_path, odd_data)