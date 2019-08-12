import sys
import mne
from mne.baseline import rescale
from tools import files
import numpy as np
import pandas as pd
import os.path as op
from scipy.signal import gaussian
from tqdm import tqdm
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
output_dir = "/cubric/scratch/c1557187/act_mis/RESULTS/THESIS_ANALYSIS"

subjects = files.get_folders_files(
    path,
    wp=False
)[0]
subjects.sort()

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
# np.convolve(data, window, mode='full')

def conv(x):
    return np.convolve(x, window, mode="full")

# data = np.apply_along_axis(conv, axis=1, arr=data)
times = np.linspace(-0.5, 2.6, num=776)

data = rescale(data, times, (-0.1, 0.0), mode="mean")

all_labels = np.array(beh.obs_dir_mod)

odd_ix = np.where(all_labels == -1)[0]
reg_ix = np.where(all_labels == 1)[0]

samples = []
for i in range(20):
    samples.append(np.random.choice(reg_ix, odd_ix.shape[0]))

sample_ix = np.hstack([odd_ix, samples[0]])

scores_all = []
for sample_reg in tqdm(samples):
    # data + labels
    sample_ix = np.hstack([odd_ix, sample_reg])
    labels = all_labels[sample_ix]
    X = data[sample_ix]

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
    scores_all.append(scores)

scores_all = np.vstack(scores_all)

scores_path = op.join(
    output_dir,
    "reg_vs_odd_svm-{}.npy".format(subject)
)

np.save(scores_path, scores_all)

print("saved")