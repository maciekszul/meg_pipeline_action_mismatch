import mne
from surfer import Brain
import numpy as np
import pandas as pd
from operator import itemgetter

# lh_ctab_path = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/lh.hcp-mmp-b_colortab.txt"
# rh_ctab_path = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/rh.hcp-mmp-b_colortab.txt"
# lh_annot_path = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/lh.hcp-mmp-b.annot"
# rh_annot_path = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/rh.hcp-mmp-b.annot"
# lut_path = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/LUT_hcp-mmp-b.txt"

# lh_ctab_path = "/cubric/data/c1557187/atlas_data/hcp-mmp/lh.hcp-mmp_colortab.txt"
# rh_ctab_path = "/cubric/data/c1557187/atlas_data/hcp-mmp/rh.hcp-mmp_colortab.txt"
# lh_annot_path = "/cubric/data/c1557187/atlas_data/hcp-mmp/lh.hcp-mmp.annot"
# rh_annot_path = "/cubric/data/c1557187/atlas_data/hcp-mmp/rh.hcp-mmp.annot"
# lut_path = "/cubric/data/c1557187/atlas_data/hcp-mmp/LUT_hcp-mmp.txt"

s_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
s_id = "fsaverage"
h = "lh"

labels = mne.read_labels_from_annot(
    subjects_dir=s_dir, 
    parc="hcp-mmp-b",
    hemi=h, 
    subject=s_id
)

lut = pd.read_csv(lut_path, sep="\t", header=None, names=["id", "file"])

lh_ctab = pd.read_csv(lh_ctab_path, sep="\t| ", header=None, engine='python')
lh_ctab = lh_ctab.drop(columns=[0, 1])
lh_ctab.columns = ["name", "R", "G", "B", "opacity"]

brain = Brain(
    subject_id="fsaverage",
    subjects_dir=s_dir,
    hemi=h,
    surf="inflated"
)

brain.add_annotation(lh_annot_path)

# for i in labels:
#     print("__________________")
#     print("currently:", i.name)
#     r, g, b = list(np.int_(np.array(i.color)[:-1] * 255))
#     x_name = lh_ctab.loc[(lh_ctab.R == r) & (lh_ctab.G == g) & (lh_ctab.B == b)].name.values[0]
#     print("should be:", x_name)
#     print("the same", i.name[:-3] == x_name)

# names = [i.name for i in labels]

aud_label = [label for label in labels if label.name == 'L_3a_ROI-lh'][0]
brain.add_label(aud_label)
aud_label = [label for label in labels if label.name == 'L_3b_ROI-lh'][0]
brain.add_label(aud_label)
aud_label = [label for label in labels if label.name == 'L_4_ROI-lh'][0]
brain.add_label(aud_label)
# lab_ix = [names.index(i) for i in names if "L" in i]
# lab_ix.extend([names.index(i) for i in names if "L_7" in i])
# lab_ix.extend([names.index(i) for i in names if "L_8" in i])
# lab_ix.extend([names.index(i) for i in names if "L_12" in i])
# lab_to_show = list(itemgetter(*lab_ix)(labels))

# for i in lab_to_show:
#     brain.add_label(i)
#     print(i.name)