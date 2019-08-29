import sys
import mne
from mne.baseline import rescale
import matplotlib.pylab as plt
import seaborn as sns
from tools import files
import numpy as np
import pandas as pd
import os.path as op
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score,
                             make_scorer)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (StratifiedKFold,
                                    train_test_split)
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

times = np.linspace(-0.6, 2.6, num=801)
data = rescale(data, times, (-0.6, -0.5), mode="mean")

data[:,:,525:] = rescale(data[:,:,525:], times[525:], (1.5, 1.6), mode="mean")

data = data[:,:, [200, 300, 400]]

all_labels = np.array(beh.obs_dir_mod)

odd_ix = np.where(all_labels == -1)[0]
reg_ix = np.where(all_labels == 1)[0]

sample_reg = np.random.choice(reg_ix, odd_ix.shape[0])

sample_ix = np.hstack([odd_ix, sample_reg])

labels = all_labels[sample_ix]
X = data[sample_ix]


# parameters for the classification
k_folds = 1 # cv folds
var_exp = 0.99 # percentage of variance
scores_all = [] # score container

for i in tqdm(range(1000)):
    labels_shuffled = np.random.permutation(labels)
    # generate iterator for cross validation
    kf = StratifiedKFold(n_splits=2, shuffle=True)
    cv_iter = kf.split(np.zeros(X.shape), labels_shuffled)

    # pipeline for classification
    cl = make_pipeline(
        RobustScaler(), 
        PCA(n_components=var_exp), 
        LinearSVC(max_iter=10000, dual=False, penalty="l1")
    )

    # temporal generalisation
    temp_genr = SlidingEstimator(
        cl, 
        n_jobs=1, 
        scoring="roc_auc"
    )

    # cross validation
    scores = cross_val_multiscore(temp_genr, X, labels_shuffled, cv=cv_iter, n_jobs=-1)
    scores_all.append(scores)



scores_all = np.vstack(scores_all)

scores_path = op.join(
    output_dir,
    "reg_vs_odd_chance_level-{}.npy".format(subject)
)

np.save(scores_path, scores_all)

print("saved")

scores_all = scores_all[::2,:]

fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(scores_all[:,0], kde=False, ax=ax, bins=50)
ax.axvline(0.5, color="red", linewidth=2)
ax.axvline(np.mean(scores_all[:,0]), color="blue", linewidth=1)
ax.set_title("1000 x shuffle the labels\n Time: 0.2 s, red line = 0.5, blue line = {}".format(np.mean(scores_all[:,0])))

fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(scores_all[:,0], kde=False, ax=ax, bins=50)
ax.axvline(0.5, color="red", linewidth=2)
ax.axvline(np.mean(scores_all[:,1]), color="blue", linewidth=1)
ax.set_title("1000 x shuffle the labels\n Time: 0.6 s, red line = 0.5, blue line = {}".format(np.mean(scores_all[:,1])))

fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(scores_all[:,2], kde=False, ax=ax, bins=50)
ax.axvline(0.5, color="red", linewidth=2)
ax.axvline(np.mean(scores_all[:,2]), color="blue", linewidth=1)
ax.set_title("1000 x shuffle the labels\n Time: 1.0 s, red line = 0.5, blue line = {}".format(np.mean(scores_all[:,2])))