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
from mne.decoding import (SlidingEstimator, 
                          GeneralizingEstimator,
                          cross_val_multiscore, 
                          get_coef)

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
meg_files = files.get_files(
    meg_path,
    "epochs-TD",
    "-epo.fif"
)[2]
meg_files.sort()

# read data

beh = pd.read_pickle(beh_file)

data = [mne.read_epochs(i) for i in meg_files]
data = np.vstack([i.pick_types(ref_meg=False).get_data() for i in data])

size, scale = 21, 2
window = gaussian(size, scale)
window = window / np.sum(window)

# def conv(x):
#     return np.convolve(x, window, mode="full")

# data = np.apply_along_axis(conv, axis=1, arr=data)

times = np.linspace(-0.6, 2.6, num=801)
data = rescale(data, times, (-0.6, -0.5), mode="mean")

data[:,:,525:] = rescale(data[:,:,525:], times[525:], (1.5, 1.6), mode="mean")

labels = np.array(beh.movement_dir_sign)

# data + labels

# parameters for the classification
k_folds = 10 # cv folds
var_exp = 0.99  # percentage of variance

# generate iterator for cross validation
kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
cv_iter = kf.split(np.zeros(data.shape), labels)

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
scores = cross_val_multiscore(temp_genr, data, labels, cv=cv_iter, n_jobs=-1)


scores_path = op.join(
    output_dir,
    "clk_vs_anti_new_baseline-{}.npy".format(subject)
)

np.save(scores_path, scores)

print("saved", scores_path)