import mne
import numpy as np
import pandas as pd


raw_path = "/cubric/scratch/c1557187/act_mis/MEG/0002/new_v1/80-000-raw.fif"
events_path = "/cubric/scratch/c1557187/act_mis/MEG/0002/new_v1/events-000-eve.fif"
beh_path = "/cubric/scratch/c1557187/act_mis/BEH/0002/ses1_0002_1558538484.csv"

raw = mne.io.read_raw_fif(raw_path)
events = mne.read_events(events_path)
beh = pd.read_csv(beh_path)
beh = beh.loc[(beh.obs_dir_mod != 0)]

onsets = mne.pick_events(events, include=[30, 40])
ends = mne.pick_events(events, include=[60, 70])
duration = ends[:,0] - onsets[:,0]

all_epochs = []
for ix, event in enumerate(onsets):
    print(ix, event[2], beh.obs_dir_mod.values[ix])
    epoch = mne.Epochs(
        raw,
        events=[event],
        baseline=None,
        preload=True,
        tmin=-0.5,
        tmax=duration[ix] / raw.info["sfreq"] + 1.1,
        detrend=1
    )
    data = epoch.get_data()[0]
    del_ints = np.arange(500, duration[ix] + 100)
    data = np.delete(data, del_ints, axis=1)
    data = data[:,:776]
    info =epoch.info
    epoch = mne.EpochsArray(
        np.array([data]),
        info,
        events=np.array([event]),
        tmin=-0.5,
        baseline=None
    )
    all_epochs.append(epoch)

epochs = mne.concatenate_epochs(all_epochs, add_offset=False)

print(np.average(onsets == epochs.events))


# tr_start = np.where((events[:,2] < 50))[0]
# tr_end = np.where((events[:,2] > 50) & (events[:,2] < 80))[0]

# tr_onsets, tr_ends = events[tr_start][:,0], events[tr_end][:,0]

# tr_durations = tr_ends - tr_onsets

# tr_cutouts = np.int_(tr_durations - 1.5*250)

# sel_evts = events[(events[:,2] == 30) | (events[:,2] == 40)]
# # TF epochs
# epochs_array = []
# for ix, i in enumerate(tr_durations):
#     epochs = mne.Epochs(
#         raw,
#         events=sel_evts[ix:ix+1],
#         baseline=None,
#         preload=True,
#         tmin=-0.5,
#         tmax=i/250 + 1.11,
#         detrend=1
#     )

#     info_save = epochs.info

#     data = epochs.get_data()[0]
#     del_ints = np.arange(500, 500 + tr_cutouts[ix])
#     data = np.delete(data, del_ints, axis=1)
#     data = data[:,:776]

#     epochs_array.append(data)

# epochs = mne.EpochsArray(
#     np.array(epochs_array),
#     epochs.info,
#     events=events[tr_start],
#     tmin=-0.5,
#     baseline=None
# )
# print(epochs)
# epochs.save(epochs_TF)

# del epochs
