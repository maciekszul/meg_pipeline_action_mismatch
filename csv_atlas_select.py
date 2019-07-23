import mne
from surfer import Brain
import numpy as np
import pandas as pd
from operator import itemgetter

atlas_csv = "/cubric/data/c1557187/meg_pipeline_action_mismatch/tools/atlas_glasser_2016.csv"

s_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
s_id = "fsaverage"
h = "both"

labels = mne.read_labels_from_annot(
    subjects_dir=s_dir, 
    parc="hcp-mmp-b",
    hemi=h, 
    subject=s_id
)

atlas_labels = pd.read_csv(atlas_csv)

primary_visual = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 1)
].LABEL_NAME.values)

early_visual = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 2)
].LABEL_NAME.values)

dorsal_stream_visual = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 3)
].LABEL_NAME.values)

ventral_stream_visual = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 4)
].LABEL_NAME.values)

mt_neighbor_visual = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 5)
].LABEL_NAME.values)

sens_mot = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 6)
].LABEL_NAME.values)

supp_sens_mot = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 7)
].LABEL_NAME.values)

pre_mot = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 8)
].LABEL_NAME.values)

insular = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 12)
].LABEL_NAME.values)

sup_parietal = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 16)
].LABEL_NAME.values)

dlpfc = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 22)
].LABEL_NAME.values)

visual_ = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 1) |
    (atlas_labels.LEVEL_1 == 2) |
    (atlas_labels.LEVEL_1 == 5)
].LABEL_NAME.values)

motor_ = list(atlas_labels.loc[
    (atlas_labels.LEVEL_1 == 6) |
    (atlas_labels.LEVEL_1 == 7)
].LABEL_NAME.values)

lh_annot_path = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/lh.hcp-mmp-b.annot"

label_list = visual_
label_list = ["L_"+ i for i in label_list] + ["R_"+ i for i in label_list]
names = [i.name[:-7] for i in labels]
picked_indices = [names.index(i) for i in label_list]
picked_labels = list(itemgetter(*picked_indices)(labels))

brain = Brain(
    subject_id="fsaverage",
    subjects_dir=s_dir,
    hemi=h,
    surf="inflated"
)

brain.add_annotation(
    lh_annot_path
)

for i in picked_labels:
    brain.add_label(i)
