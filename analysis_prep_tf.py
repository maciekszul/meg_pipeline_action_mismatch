import pandas as pd
from scipy.signal import hilbert
from mne.filter import filter_data
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

# try:
#     key_freq = str(sys.argv[2])
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

freq_dict = {
    "alpha": (7, 14),
    "beta": (14, 30),
    "delta": (None, 4),
    "theta": (4, 7),
    "low_gamma": (30, 80)
}

# low_freq, high_freq = freq_dict[key_freq]



################################################################################

# subject = subjects[index]

for key_freq in freq_dict.keys():
    low_freq, high_freq = freq_dict[key_freq]
    print(low_freq, high_freq)

    for subject in tqdm(subjects[:-4]):
        print(subject)
        trials_all = files.get_files(path, subject, ".npy")[2]
        trials_all.sort()

        enumerator = list(zip(range(len(trials_all)), trials_all))

        data_dict = dict()
        def extract(input_):
            ix, path_to_a_file = input_
            data = np.load(path_to_a_file).item()
            data = np.array([data[key].reshape(-1) for key in label_list])
            data = filter_data(
                data,
                sfreq=250,
                l_freq=low_freq,
                h_freq=high_freq,
                method="fir",
                phase="minimum",
                n_jobs=1
            )

            # hilbert envelope 
            data = np.abs(hilbert(data, axis=0))
            # baseline
            data = data - np.mean(data[:,:125], axis=1)[:, np.newaxis]
            data_dict[ix] = data

        # extract(enumerator[0])

        Parallel(n_jobs=-1, prefer="threads")(
            delayed(extract)(input_) for input_ in enumerator)

        meg_file = op.join(
            path_out,
            "{}-{}.npy".format(key_freq, subject)
        )

        np.save(meg_file, data_dict)
