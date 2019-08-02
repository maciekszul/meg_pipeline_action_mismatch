import mne
import numpy as np

raw_path = "/cubric/scratch/c1557187/act_mis/MEG/0002/220519-501_371SensMotTrans_20190522_04.ds"


raw = mne.io.read_raw_ctf(
    raw_path,
    system_clock="ignore",
    clean_names=True,
    preload=True
)

set_ch = {"EEG057":"eog", "EEG058": "eog", "UPPT001": "stim"}
raw.set_channel_types(set_ch)

events_crop = mne.find_events(
    raw
)

events = mne.find_events(
    raw
)

crop_min = events_crop[0][0]/ 1200 - 5
crop_max = events_crop[-1][0]/ 1200 + 5
raw.crop(tmin=crop_min, tmax=crop_max)

raw = raw.pick_types(
    meg=True,
    ref_meg=True,
    eog=True,
    eeg=False,
    stim=True
)

filter_picks = mne.pick_types(
    raw.info,
    meg=True,
    ref_meg=True,
    stim=False,
    eog=True
)

raw = raw.filter(
    0.1,
    None,
    method="fir",
    phase="minimum",
    n_jobs=-1,
    picks=filter_picks
)

raw = raw.filter(
    None,
    80,
    method="fir",
    phase="minimum",
    n_jobs=-1,
    picks=filter_picks
)

raw, events = raw.copy().resample(
    250, 
    npad="auto", 
    events=events,
    n_jobs=-1,
)

np.hstack([events_crop[:,0][:, np.newaxis]/1200, events[:,0][:, np.newaxis]/250])
np.hstack([events_crop[:,2][:, np.newaxis], events[:,2][:, np.newaxis]])

raw.save("test-raw.fif", overwrite=True)