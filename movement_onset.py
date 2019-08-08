import mne
import numpy as np
import pandas as pd

raw_path = "/cubric/scratch/c1557187/act_mis/MEG/0002/new_v1/80-000-raw.fif"
events_path = "/cubric/scratch/c1557187/act_mis/MEG/0002/new_v1/events-000-eve.fif"
beh_path = "/cubric/scratch/c1557187/act_mis/RESULTS/BEH_PARTICIPANT/beh-0002.gz"

# read data 

beh = pd.read_pickle(beh_path)
beh = beh.loc[(beh.exp_type == "1")]

rotation_onsets = mne.read_events(
    events_path,
    include=[30,40]
)

blink_onsets = mne.read_events(
    events_path,
    include=[60,70]
)

# raw = mne.io.read_raw_fif(
#     raw_path,
#     preload=True
# )

all_epochs = []
for ix, event in enumerate(rotation_onsets):
    print(event[0], blink_onsets[ix][0], blink_onsets[ix][0]-event[0], blink_onsets[ix][0]-event[0]-beh.action_onset[ix]-250)
    event[0] += beh.action_onset[ix]
    epoch = mne.Epochs(
        raw,
        events=[event],
        baseline=None,
        preload=True,
        tmin=-0.5,
        tmax=1.0,
        detrend=1
    )
#     data = epoch.get_data()[0]
#     del_ints = np.arange(500, duration[ix] + 100)
#     data = np.delete(data, del_ints, axis=1)
#     data = data[:,:776]
#     info =epoch.info
#     epoch = mne.EpochsArray(
#         np.array([data]),
#         info,
#         events=np.array([event]),
#         tmin=-0.5,
#         baseline=None
#     )
#     all_epochs.append(epoch)

# epochs = mne.concatenate_epochs(all_epochs, add_offset=False)