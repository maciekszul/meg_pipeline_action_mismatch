import mne
from tools import files
import numpy as np
import pandas as pd


subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
label_path = "tools/label"

lh_labels = files.get_files(label_path, "lh", ".label")[2]
rh_labels = files.get_files(label_path, "rh", ".label")[2]
lh_annot = files.get_files("tools", "lh", ".annot")[2][0]
rh_annot = files.get_files("tools", "rh", ".annot")[2][0]

stc_path = "/cubric/scratch/c1557187/act_mis/RESULTS/WHATEVER_SOURCE_SPACE/reg-001-0043"



stc = mne.read_source_estimate(stc_path)

