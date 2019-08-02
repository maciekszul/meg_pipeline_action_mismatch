import pandas as pd
import numpy as np
import os.path as op
from tools import files
import mne
import sys
import matplotlib.pylab as plt
import itertools as it

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    file_index = int(sys.argv[2])
except:
    print("incorrect arguments")
    sys.exit()

beh_path = "/cubric/data/c1557187/meg_pipeline_action_mismatch/beh_data/all_trials.pkl"
subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
path = "/cubric/scratch/c1557187/act_mis/RESULTS/TF_SOURCE_SPACE_PROC"
data_path = "/cubric/scratch/c1557187/act_mis"

subjects = files.get_folders_files(subjects_dir, wp=False)[0]
subjects.sort()

subject = subjects[index]

subject_meg = op.join(
        data_path,
        "MEG",
        subject
    )

subject_meg_meg = op.join(
        data_path,
        "MEG",
        subject,
        "new_v1"
    )

subject_beh = op.join(
    data_path,
    "BEH",
    subject
)

beh_files = files.get_files(subject_beh, "ses", "csv")[2]
beh_files.sort()
beh = pd.read_csv(beh_files[file_index])

event_files = files.get_files(subject_meg_meg, "events", "-eve.fif")[2]
event_files.sort()
events_from_file = mne.read_events(event_files[file_index], include=[30,40,50])

raw_files = files.get_folders_files(subject_meg)[0]
raw_files.sort()


raw = mne.io.read_raw_ctf(
    raw_files[file_index],
    preload=True
)

events = mne.find_events(
    raw,
    stim_channel="UPPT001",
    min_duration=0.003
)

rot_events = mne.pick_events(
    events,
    include=[30, 40, 50]
)

obs_events = mne.pick_events(
    events,
    include=[60, 70, 80]
)

mov_dir = beh.obs_dir_mod.values[:, np.newaxis]
rot_dir = rot_events[:,2][:, np.newaxis]
eve_dir = events_from_file[:,2][:, np.newaxis]
eve_time_diff = np.insert(np.diff(events_from_file[:,0])/250, 0, 0.0)[:, np.newaxis]
rot_time_diff = np.insert(np.diff(rot_events[:,0])/1200, 0, 0.0)[:, np.newaxis]

print(np.hstack([mov_dir, rot_dir, eve_dir, rot_time_diff, eve_time_diff]))