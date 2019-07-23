import numpy as np
import pandas as pd
import sys
from tools import files
import mne
import os.path as op
from operator import itemgetter
import matplotlib.pylab as plt
from scipy.stats import trim_mean
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    subj_index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()


# paths
atlas_csv = "/cubric/data/c1557187/meg_pipeline_action_mismatch/tools/atlas_glasser_2016.csv"
subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
main_path = "/cubric/scratch/c1557187/act_mis"

subjects = files.get_folders_files(subjects_dir, wp=False)[0]
subjects.sort()
subject = subjects[subj_index]

print("subject:", subject)

subject_meg = op.join(
    main_path,
    "MEG",
    subject,
    "new_v1"
)

# functions
trim = lambda x: trim_mean(x, 0.1, axis=0)

# Glasser et al. 2016 labels
s_id = "fsaverage"
h = "both"

labels = mne.read_labels_from_annot(
    subjects_dir=subjects_dir, 
    parc="hcp-mmp-b",
    hemi=h, 
    subject=s_id
)

atlas_labels = pd.read_csv(atlas_csv)

visual_ = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 1) |
    (atlas_labels.LEVEL_1 == 2) |
    (atlas_labels.LEVEL_1 == 5)
].LABEL_NAME.values)

motor_ = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 6) |
    (atlas_labels.LEVEL_1 == 7)
].LABEL_NAME.values)

label_list = visual_

label_list = ["L_"+ i for i in label_list] + ["R_"+ i for i in label_list]
names = [i.name[:-7] for i in labels]
picked_indices = [names.index(i) for i in label_list]
picked_labels = list(itemgetter(*picked_indices)(labels))

# files read and

all_epochs = files.get_files(subject_meg, "epochs-TD", "-epo.fif")[2]
all_epochs.sort()
inv_sol = files.get_files(subject_meg, "epochs-TD", "-inv.fif")[2]
inv_sol.sort()


morph = mne.read_source_morph(
    op.join(
        subject_meg,
        "{}-morph.h5".format(subject)
    )
)

label_timecourse = {i: [] for i in label_list}

all_files = list(zip(inv_sol, all_epochs))
for inv_path, epo_path in all_files:
    epochs = mne.read_epochs(epo_path, preload=True)
    epochs = epochs.apply_baseline((-0.1, 0.0))
    epochs = epochs.apply_baseline((1.6, 2.6))
    reg_avg = epochs["30"].average(method=trim)
    odd_avg = epochs["40"].average(method=trim)
    diff_avg = mne.combine_evoked([odd_avg, -reg_avg], weights="equal")
    
    inverse_operator = mne.minimum_norm.read_inverse_operator(
        inv_path
    )

    del epochs

    diff_stc = mne.minimum_norm.apply_inverse(
        diff_avg,
        inverse_operator,
        lambda2= 1/3**2,
        method="dSPM"
    )
    diff_stc = morph.apply(diff_stc)
    times = diff_stc.times
    for i in tqdm(picked_labels):
        label_timecourse[i.name[:-7]].append(diff_stc.in_label(i).data)

for key in tqdm(label_timecourse.keys()):
    data = np.array(label_timecourse[key])
    data = np.average(data, axis=0)
    pca = PCA(n_components=1)
    data = pca.fit_transform(data.transpose())
    label_timecourse[key] = data

output = op.join(
    main_path,
    "RESULTS",
    "TD_SOURCE_SPACE_AVG",
    "{}-td-visual-cortex.npy".format(subject)
)

np.save(output, label_timecourse)