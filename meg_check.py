import sys
import os.path as op
import numpy as np
import pandas as pd
from tools import files

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

data_path = "/cubric/scratch/c1557187/act_mis/RESULTS/TF_SOURCE_SPACE"

beh_files = files.get_files(data_path, "beh", ".gz")[2]
beh_files.sort()
subjects = [i.split("-")[-1].split(".")[0] for i in beh_files]
subjects.sort()

subject = subjects[index]
subject_files = files.get_files(
    data_path,
    subject,
    ".npy"
)[2]
subject_files.sort()

cond = np.array([i.split("/")[-1].split(".")[0].split("-") for i in subject_files])
beh = pd.read_pickle(beh_files[index])

print(np.hstack([np.array(beh.obs_dir_mod.values)[:,np.newaxis], cond[:,3][:,np.newaxis]]))