import json
import os.path as op
import numpy as np
import pandas as pd
# from mne.stats import *
from scipy.stats import trim_mean
from tools import files
from tqdm import tqdm


data_path = "/cubric/scratch/c1557187/act_mis/RESULTS/TD_SOURCE_SPACE_PROC"
label_json = op.join(data_path, "label_order.json")
labels_csv = "/cubric/data/c1557187/meg_pipeline_action_mismatch/tools/atlas_glasser_2016.csv"

which_freq = "low_gamma"

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

particpant_beh_files = files.get_files(
    data_path,
    "beh",
    ".gz"
)[2]
particpant_beh_files.sort()

particpant_meg_files = files.get_files(
    data_path,
    which_freq,
    ".npy"
)[2]
particpant_meg_files.sort()

pp_list = [i.split("/")[-1].split("-")[-1].split(".")[0] for i in particpant_beh_files]

zip_files = list(zip(pp_list, particpant_beh_files, particpant_meg_files))

odd_means = dict()
reg_means = dict()

for pp, beh_path, meg_path in tqdm((zip_files)):
    print(pp)

    beh = pd.read_pickle(beh_path)
    meg = np.load(meg_path).item()

    keys_odd = beh.loc[(beh.obs_dir_mod == -1)].index.values
    keys_reg = beh.loc[(beh.obs_dir_mod == 1)].index.values

    odd_means[pp] = trim_mean(np.array([meg[key] for key in keys_odd]), 0.1, axis=0)
    reg_means[pp] = trim_mean(np.array([meg[key] for key in keys_reg]), 0.1, axis=0)

    del meg

odd_file = op.join(data_path, "means-{}-odd.npy".format(which_freq))
reg_file = op.join(data_path, "means-{}-reg.npy".format(which_freq))

np.save(odd_file, odd_means)
np.save(reg_file, reg_means)