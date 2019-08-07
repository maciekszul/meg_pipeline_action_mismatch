import pandas as pd
from scipy.signal import hilbert
from mne.filter import filter_data
from mne.baseline import rescale
import numpy as np
import os.path as op
from tools import files
import sys
from joblib import Parallel, delayed
from tqdm import tqdm

# try:
#     index = int(sys.argv[1])
# except:
#     print("incorrect arguments")
#     sys.exit()

path = "/cubric/scratch/c1557187/act_mis/RESULTS/TF_SOURCE_SPACE"
path_out = "/cubric/scratch/c1557187/act_mis/RESULTS/TD_SOURCE_SPACE_PROC"
subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
beh_path = "/cubric/data/c1557187/meg_pipeline_action_mismatch/beh_data/all_trials.pkl"

subjects = files.get_folders_files(subjects_dir, wp=False)[0]
subjects.sort()

atlas_csv = "/cubric/data/c1557187/meg_pipeline_action_mismatch/tools/atlas_glasser_2016.csv"
atlas_labels = pd.read_csv(atlas_csv)
label_list = atlas_labels.LABEL_NAME.values.tolist()
left = ["L_" + i for i in label_list]
right = ["R_" + i for i in label_list]
label_list = left + right
label_list.sort()

################################################################################

# subject = subjects[index]
for subject in tqdm(subjects[:-4]):
    trials_all = files.get_files(path, subject, ".npy")[2]
    trials_all.sort()

    times = np.linspace(-0.5, 2.6, num=776)

    enumerator = list(zip(range(len(trials_all)), trials_all))

    data_dict = dict()
    def extract(input_):
        ix, path_to_a_file = input_
        data = np.load(path_to_a_file).item()
        data = np.array([data[key].reshape(-1) for key in label_list])
        data = filter_data(
            data,
            sfreq=250,
            l_freq=None,
            h_freq=30,
            method="fir",
            phase="minimum",
            n_jobs=1
        )
        # # baseline
        data = rescale(data, times, (-0.1, 0.0), mode="mean")
        data = rescale(data, times, (1.6, 2.6), mode="mean")
        data_dict[ix] = data


    Parallel(n_jobs=-1, prefer="threads")(
        delayed(extract)(input_) for input_ in enumerator)

    meg_file = op.join(
        path_out,
        "meg-{}.npy".format(subject)
    )

    np.save(meg_file, data_dict)
