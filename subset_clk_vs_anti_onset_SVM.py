import sys
import mne
from mne.baseline import rescale
from tools import files
import numpy as np
import pandas as pd
import os.path as op
from scipy.signal import gaussian
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from mne.decoding import (
    SlidingEstimator, 
    GeneralizingEstimator,
    cross_val_multiscore, 
    get_coef
)

try:
    range_index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

# get paths

path = "/cubric/scratch/c1557187/act_mis/MEG"
beh_path = "/cubric/scratch/c1557187/act_mis/RESULTS/BEH_PARTICIPANT"
output_dir = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS/SVM"

subjects = files.get_folders_files(
    path,
    wp=False
)[0]
subjects.sort()

remove_subj = [7, 32, 34, 35, 38, 40, 41, 43]
remove_subj = [str(i).zfill(4) for i in remove_subj]
[subjects.remove(i) for i in remove_subj]
subject = subjects[range_index]

print(subject)

beh_file = files.get_files(
    beh_path,
    "beh-{}".format(subject),
    ".gz"
)[2][0]

meg_path = op.join(
    path,
    subject,
    "new_v1"
)

onset_files = files.get_files(
    meg_path,
    "epochs-resp-TD",
    "-epo.fif"
)[2]
onset_files.sort()

obs_files = files.get_files(
    meg_path,
    "epochs-TD",
    "-epo.fif"
)[2]
obs_files.sort()

onset_times = np.linspace(-0.5, 1, num=376)
obs_times = np.linspace(-0.5, 2.6, num=776)

# prepare the sensor groupings the subset

info = mne.io.read_info(onset_files[0])
ch_subset = mne.pick_types(info, ref_meg=False, eog=False, stim=False)
info = mne.pick_info(info, ch_subset)

sensor_groupings = {
    "A": [i for i in info["ch_names"] if "MLO" in i],
    "B": [i for i in info["ch_names"] if "MRO" in i],
    "C": [i for i in info["ch_names"] if "MLC" in i],
    "D": [i for i in info["ch_names"] if "MRC" in i],
    "E": [i for i in info["ch_names"] if "0" in i],
    "F": [i for i in info["ch_names"] if "C" in i]
}

# read data
beh = pd.read_pickle(beh_file)

onsets = [mne.read_epochs(i) for i in onset_files]
onsets = np.vstack([i.pick_types(ref_meg=False).get_data() for i in onsets])

onsets = rescale(onsets, onset_times, (-0.5, 0.4), mode="mean")

obs = [mne.read_epochs(i) for i in obs_files]
obs = np.vstack([i.pick_types(ref_meg=False).get_data() for i in obs])

obs = rescale(obs, obs_times, (1.6, 2.6), mode="mean")
obs = rescale(obs, obs_times, (1.5, 1.6), mode="mean")

data = np.concatenate([onsets, obs[:,:,500:]], axis=-1)

labels = np.array(beh.movement_dir_sign)

# keys loop
for key in sensor_groupings.keys():
    ch_selection = mne.pick_channels(info["ch_names"], sensor_groupings["D"])

    X = data[:, ch_selection, :]

    # parameters for the classification
    k_folds = 10 # cv folds
    var_exp = 0.99  # percentage of variance

    # generate iterator for cross validation
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    cv_iter = kf.split(np.zeros(X.shape), labels)

    # pipeline for classification
    cl = make_pipeline(
        RobustScaler(), 
        PCA(n_components=var_exp), 
        LinearSVC(max_iter=10000, dual=False, penalty="l1")
    )

    # temporal generalisation
    temp_genr = GeneralizingEstimator(
        cl, 
        n_jobs=1, 
        scoring="roc_auc"
    )

    # cross validation
    scores = cross_val_multiscore(temp_genr, X, labels, cv=cv_iter, n_jobs=-1)

    scores_path = op.join(
        output_dir,
        "{}-clk_vs_anti_onset-{}.npy".format(key, subject)
    )

    np.save(scores_path, scores)

    print("saved")