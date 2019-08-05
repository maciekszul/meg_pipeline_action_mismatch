import json
import os.path as op
import numpy as np
import pandas as pd
from mne.stats import *
from scipy.stats import trim_mean, sem
from tools import files
import matplotlib.pylab as plt

data_path = "/cubric/scratch/c1557187/act_mis/RESULTS/TD_SOURCE_SPACE_PROC"
label_json = op.join(data_path, "label_order.json")
labels_csv = "/cubric/data/c1557187/meg_pipeline_action_mismatch/tools/atlas_glasser_2016.csv"

which_freq = "meg"

freq_dict = {
    "alpha": (7, 14),
    "beta": (14, 30),
    "delta": (None, 4),
    "theta": (4, 7),
    "low_gamma": (30, 80)
}

with open(label_json) as json_file:
    labels = json.load(json_file)

label_data = pd.read_csv(labels_csv)

mean_files = files.get_files(
    data_path,
    "means-{}".format(which_freq),
    ".npy"
)[2]
mean_files.sort()

odd, reg = [np.load(i).item() for i in mean_files]

odd_mean = np.mean(np.array([odd[key] for key in odd.keys()]), axis=0)
odd_sem = sem(np.array([odd[key] for key in odd.keys()]), axis=0)

reg_mean = np.mean(np.array([reg[key] for key in reg.keys()]), axis=0)
reg_sem = sem(np.array([reg[key] for key in reg.keys()]), axis=0)
