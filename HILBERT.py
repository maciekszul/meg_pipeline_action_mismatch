import sys
import numpy as np
import pandas as pd
import mne
import json
import os.path as op
from tools import files
from tqdm import tqdm
import matplotlib.pylab as plt


try:
    index = int(sys.argv[1])
except:
    print("incorrect file index")
    sys.exit()

try:
    file_index = int(sys.argv[2])
except:
    print("incorrect file index")
    sys.exit()

data_path = "/cubric/scratch/c1557187/act_mis"
path = "/cubric/scratch/c1557187/act_mis/MEG"
output_dir = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS/HILBERT"

subjects = files.get_folders_files(
    path,
    wp=False
)[0]
subjects.sort()

subject = subjects[index]

subject_meg = op.join(
    path,
    subject,
    "new_v1"
)

event_files = files.get_files(
    subject_meg,
    "events",
    "-eve.fif"
)[2]
event_files.sort()

raw_files = files.get_files(
    subject_meg,
    "80",
    "-raw.fif"
)[2]
raw_files.sort()

ica_files = files.get_files(
    subject_meg,
    "80-",
    "-ica.fif"
)[2]
ica_files.sort()

beh_proc = op.join(
    data_path,
    "RESULTS",
    "BEH_PARTICIPANT"
)
beh_file = files.get_files(
    beh_proc,
    "beh-{}".format(subject),
    ".gz"
)[2][0]

beh = pd.read_pickle(beh_file)

components_file_path = op.join(
    subject_meg,
    "rejected-components.json"
)
with open(components_file_path) as data:
    components_rej = json.load(data)

raw_file = raw_files[file_index]
event_file = event_files[file_index]
ica_file = ica_files[file_index]
key_file = raw_file.split("/")[-1]
file_no = key_file.split("-")[1]

print(subject)
print(raw_file)
print(event_file)
print(ica_file)
print(key_file)
print(file_no)

events = mne.read_events(
    event_file
)

freq_bands = {
    "alpha": (9, 14),
    "beta": (14, 30),
    "theta": (4, 7),
    "low_gamma": (30, 80),
    "stimulus": (7, 9)
}

raw = mne.io.read_raw_fif(
    raw_file,
    preload=True
)

ica = mne.preprocessing.read_ica(ica_file)

raw = ica.apply(
    raw,
    exclude=components_rej[key_file]
)

filter_picks = mne.pick_types(
    raw.info,
    meg=True,
    ref_meg=True,
    stim=False,
    eog=True
)

for key in freq_bands.keys():
    l_freq, h_freq = freq_bands[key]
    print(key, l_freq, h_freq)

    raw_bp = raw.copy().filter(
        l_freq,
        h_freq,
        method="fir",
        phase="minimum",
        n_jobs=-1,
        picks=filter_picks
    )

    raw_bp.apply_hilbert(n_jobs=-1, envelope=False)

    onsets = mne.pick_events(events, include=[30, 40])
    ends = mne.pick_events(events, include=[60, 70])
    duration = ends[:,0] - onsets[:,0]
    all_epochs = []
    for ix, event in enumerate(onsets):
        epoch = mne.Epochs(
            raw_bp,
            events=[event],
            baseline=None,
            preload=True,
            tmin=-0.6,
            tmax=duration[ix] / raw.info["sfreq"] + 1.1,
            detrend=1
        )
        data = epoch.get_data()[0]
        del_ints = np.arange(525, duration[ix] + 100)
        data = np.delete(data, del_ints, axis=1)
        data = data[:,:801]
        info =epoch.info
        epoch = mne.EpochsArray(
            np.array([data]),
            info,
            events=np.array([event]),
            tmin=-0.6,
            baseline=None
        )
        all_epochs.append(epoch)

    epochs = mne.concatenate_epochs(all_epochs, add_offset=False)

    epochs.subtract_evoked()

    epochs = mne.EpochsArray(
        data=np.abs(epochs.get_data()),
        info=epochs.info,
        events=epochs.events,
        tmin=epochs.tmin
    )

    epochs_path = op.join(
        subject_meg,
        "epochs-{}-{}-epo.fif".format(key, file_no)
    )
    epochs.save(epochs_path)
    del epochs
    all_epochs = []
    for ix, event in enumerate(onsets):
        event[0] += beh.action_onset[ix]
        epoch = mne.Epochs(
            raw_bp,
            events=[event],
            baseline=None,
            preload=True,
            tmin=-1,
            tmax=1,
            detrend=1
        )
        all_epochs.append(epoch)

    epochs = mne.concatenate_epochs(all_epochs, add_offset=False)

    epochs.subtract_evoked()

    epochs = mne.EpochsArray(
        data=np.abs(epochs.get_data()),
        info=epochs.info,
        events=epochs.events,
        tmin=epochs.tmin
    )
    
    epochs_path = op.join(
        subject_meg,
        "epochs-resp-{}-{}-epo.fif".format(key, file_no)
    )

    epochs.save(epochs_path)
    del epochs
    del raw_bp
    print(key, "done")


# evoked = epochs.average()

# ci_low, ci_up = mne.stats._bootstrap_ci(
#     evoked.data, 
#     random_state=0,
#     stat_fun=lambda x: np.sum(x ** 2, axis=0)
# )

# gfp = np.sum(evoked.data ** 2, axis=0)
# gfp = mne.baseline.rescale(gfp, epochs.times, baseline=(-0.5, 0))

# ci_low = mne.baseline.rescale(ci_low, evoked.times, baseline=(-0.5, 0))
# ci_up = mne.baseline.rescale(ci_up, evoked.times, baseline=(-0.5, 0))

# plt.fill_between(epochs.times, gfp + ci_up, gfp - ci_low, alpha=0.3)
# plt.plot(epochs.times, gfp)
# plt.axhline(0)
# plt.show()