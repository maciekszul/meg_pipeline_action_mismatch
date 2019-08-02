import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)

import numpy as np
import pandas as pd
import sys
from tools import files
import mne
import os.path as op
from operator import itemgetter
from tqdm import tqdm
import matplotlib.pylab as plt
from scipy.stats import trim_mean
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

atlas_csv = "/cubric/data/c1557187/meg_pipeline_action_mismatch/tools/atlas_glasser_2016.csv"
subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
main_path = "/cubric/scratch/c1557187/act_mis"


subjects = files.get_folders_files(subjects_dir, wp=False)[0]
subjects.sort()

subject = subjects[int(index//2)]

file_list = []

for sub in [subjects[0]]:
    subject_meg = op.join(
        main_path,
        "MEG",
        sub,
        "new_v1"
    )
    all_epochs = files.get_files(subject_meg, "epochs-TF", "-epo.fif")[2]
    all_epochs.sort()
    inv_sol = files.get_files(subject_meg, "epochs-TF", "-inv.fif")[2]
    inv_sol.sort()
    morph_path = op.join(subject_meg, "{}-morph.h5".format(subject))
    file_list.append(tuple(list(list(zip(inv_sol, all_epochs))[0]) + [morph_path]))
    file_list.append(tuple(list(list(zip(inv_sol, all_epochs))[1]) + [morph_path]))

inv_path, epo_path, morph_path = file_list[index]

print(inv_path)
print(epo_path)
print(morph_path)

# read files
inv = mne.minimum_norm.read_inverse_operator(inv_path)
morph = mne.read_source_morph(morph_path)
epochs = mne.read_epochs(epo_path, preload=True)
atlas_labels = pd.read_csv(atlas_csv)

# label selection

labels = mne.read_labels_from_annot(
    subjects_dir=subjects_dir, 
    parc="hcp-mmp-b",
    hemi="both", 
    subject="fsaverage"
)

# processing
trial_order = epochs.events[:,2]
names = [i.name[:-7] for i in labels]

epochs = epochs.apply_baseline((-0.1, 0.0))
epochs = epochs.apply_baseline((1.6, 2.6))

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start processing:", time_string)

evoked = epochs[0].average()

del epochs

source = mne.minimum_norm.apply_inverse(
    evoked,
    inv,
    lambda2= 1/3**2,
    method="dSPM"
)
source = morph.apply(source)

label_timecourse = {i: [] for i in names}
pca = PCA(n_components=1)


# def lab_extr_pca(source, label):
#     data = source.in_label(label).data
#     data = pca.fit_transform(data.transpose())
#     label_timecourse[label.name[:-7]] = data

# Parallel(n_jobs=-1, prefer="threads")(
#     delayed(lab_extr_pca)(source, label_) for label_ in labels[9:13])

# labs = [i.name for i in labels[9:13]]

# f = mne.filter.filter_data(label_timecourse["L_24dv_ROI-lh"], sfreq=250, l_freq=14, h_freq=30)
# f = hilbert(f)
# f = np.sqrt(f**2)