import pandas as pd
import numpy as np
import os.path as op
from tools import files
import mne
import json
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pylab as plt
import itertools as it

# try:
#     index = int(sys.argv[1])
# except:
#     print("incorrect arguments")
#     sys.exit()

beh_path = "/cubric/data/c1557187/meg_pipeline_action_mismatch/beh_data/all_trials.pkl"
label_json = "/cubric/scratch/c1557187/act_mis/RESULTS/TD_SOURCE_SPACE_PROC/label_order.json"
subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
path = "/cubric/scratch/c1557187/act_mis/RESULTS/TF_SOURCE_SPACE"
data_path = "/cubric/scratch/c1557187/act_mis"


subjects = files.get_folders_files(subjects_dir, wp=False)[0]
subjects.sort()

beh_tr = dict()
meg_tr = dict()

for index in range(24):
    subject = subjects[index]

    subject_meg = op.join(
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

    event_files = files.get_files(subject_meg, "events", "-eve.fif")[2]
    event_files.sort()
    beh_files = files.get_files(subject_beh, "ses", "csv")[2]
    beh_files.sort()

    ev = np.vstack([mne.read_events(i, include=[30, 40])for i in event_files])

    bh = []
    for i in beh_files:
        f = pd.read_csv(i)
        f = f.loc[(f.obs_dir_mod != 0)]
        f = np.array(f.obs_dir_mod.values)
        bh.append(f)
    bh = np.concatenate(bh)

    ev[:,2][ev[:,2] == 30] = 1
    ev[:,2][ev[:,2] == 40] = -1

    print(subject)

    # print(bh == ev[:,2])

    beh_tr[subject] = bh
    meg_tr[subject] = ev[:,2]


ppvspp = list(it.product(range(24), range(24)))

mx = np.zeros((24, 24))

for x, y in ppvspp:
    b = beh_tr[subjects[x]]
    m = meg_tr[subjects[y]]
    try:
        res = np.sum(b == m)/280
        if res == 1:
            res = 2
    except:
        res = 0
    mx[x, y] = res


