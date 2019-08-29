import sys
import mne
from mne.baseline import rescale
from tools import files
import numpy as np
import pandas as pd
import os.path as op
from sklearn.metrics import (accuracy_score,
                             make_scorer)
from tqdm import tqdm
# from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


times = np.linspace(-0.6, 2.6, num=801)
data = rescale(data, times, (-0.6, -0.5), mode="mean")

data[:,:,525:] = rescale(data[:,:,525:], times[525:], (1.5, 1.6), mode="mean")

all_labels = np.array(beh.obs_dir_mod)
cw_labels = beh.movement_dir_sign.values

odd_ix = np.where(all_labels == -1)[0]
reg_ix = np.where(all_labels == 1)[0]

try:
    x, (odd_cw, odd_acw) = np.unique(cw_labels[odd_ix], return_counts=True)
    odd_ratio = odd_cw / odd_acw
except ValueError:
    odd_direction, odd = np.unique(cw_labels[odd_ix], return_counts=True)
    if odd.shape[0] == 1:
        odd_cw, odd_acw, odd_ratio = [0] *3


ratio_ratio = 0
while ratio_ratio != 1:
    sample_reg_ix = np.random.choice(reg_ix, odd_ix.shape[0])
    try:
        x, (sample_reg_cw, sample_reg_acw) = np.unique(cw_labels[sample_reg_ix], return_counts=True)
        sample_reg_ratio = sample_reg_cw / sample_reg_acw
        ratio_ratio = sample_reg_ratio / odd_ratio
    except ValueError:
        sample_reg_direction, sample_reg = np.unique(cw_labels[sample_reg_ix], return_counts=True)
        if odd.shape[0] == 1:
            sample_reg_cw, sample_reg_acw, sample_reg_ratio = [0] * 3
        else:
            raise ValueError("No trials")
        if odd_direction == sample_reg_direction:
            ratio_ratio = 1
        else:
            raise ValueError("something wrong with the movement directions")

    print(ratio_ratio)

print(
    "odd_cw:",
    odd_cw, 
    "odd_acw:",
    odd_acw, 
    "odd_cw/acw_ratio:",
    odd_ratio, 
    "!", 
    "sample_reg_cw:",
    sample_reg_cw, 
    "sample_reg_acw:",
    sample_reg_acw, 
    "sample_reg_cw/acw_ratio:",
    sample_reg_ratio, 
    "!", 
    ratio_ratio
)

print("odd_labels", cw_labels[odd_ix], "mean:", np.mean(cw_labels[odd_ix]))
print("reg_labels", cw_labels[sample_reg_ix], "mean:", np.mean(cw_labels[sample_reg_ix]))

sample_ix = np.hstack([odd_ix, sample_reg_ix])

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
    LinearDiscriminantAnalysis()
)

# temporal generalisation
temp_genr = SlidingEstimator(
    cl, 
    n_jobs=1, 
    scoring=make_scorer(accuracy_score)
)

# cross validation
scores = cross_val_multiscore(temp_genr, X, labels, cv=cv_iter, n_jobs=-1)
scores_all = []
scores_all.append(scores)

scores_all = np.vstack(scores_all)

scores_path = op.join(
    output_dir,
    "reg_vs_odd_lda_balanced-{}.npy".format(subject)
)

np.save(scores_path, scores_all)

print("saved")