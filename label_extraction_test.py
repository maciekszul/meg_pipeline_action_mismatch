import mne
from tools import files
import numpy as np
import pandas as pd
from surfer import Brain
from operator import itemgetter
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
lh_annot = "//cubric/data/c1557187/meg_pipeline_action_mismatch/tools/lh.HCP-MMP1.annot"
rh_annot = "//cubric/data/c1557187/meg_pipeline_action_mismatch/tools/rh.HCP-MMP1.annot"

stc_path = "/cubric/scratch/c1557187/act_mis/RESULTS/WHATEVER_SOURCE_SPACE/dif-001-0043"

stc = mne.read_source_estimate(stc_path)

labels = mne.read_labels_from_annot(subjects_dir=subjects_dir, parc="hcp-mmp-b", hemi="lh", subject="fsaverage")
names = [i.name for i in labels]


subject_id = "fsaverage"
hemi = "lh"
surf = "inflated"

# visual
# lab_ix = [lh_names.index(i) for i in lh_names if "L_V" in i]
# lab_ix = lab_ix[:-5]
# lab_ix.append(90)
# lab_to_show = list(itemgetter(*lab_ix)(lh_labels))

# motor
lab_ix = [names.index(i) for i in names if "L_6" in i]
lab_ix.extend([names.index(i) for i in names if "L_7" in i])
lab_ix.extend([names.index(i) for i in names if "L_8" in i])
# lab_ix.extend([lh_names.index(i) for i in lh_names if "L_5" in i])

lab_to_dict = list(itemgetter(*lab_ix)(names))
lab_to_show = list(itemgetter(*lab_ix)(labels))

lab_mean = {i: None for i in lab_to_dict}

lab_pca = {i: None for i in lab_to_dict}


brain = Brain(
    subject_id=subject_id,
    subjects_dir=subjects_dir,
    hemi=hemi,
    surf=surf
)

# brain.add_annotation(lh_annot)

for i in lab_to_show:
    brain.add_label(i)
    # brain.add_foci(i.vertices[len(i.vertices)//2], coords_as_verts=True, map_surface=None, scale_factor=.5, color="red")
    print(i.name)

# for i in lab_to_show:
#     lab_mean[i.name] = np.average(stc.in_label(i).data, axis=0)
#     pca = PCA(n_components=1)
#     data_pca = pca.fit_transform(stc.in_label(i).data.transpose())
#     lab_pca[i.name] = data_pca  





# figure = plt.figure(figsize=(20, 10))
# ax = figure.add_subplot(111)
# for key in lab_mean.keys():
#     ax.plot(stc.times, lab_mean[key], label=key)
# ax.axvline(0, linewidth=0.2, color="black")
# ax.axvline(1.6, linewidth=0.2, color="black")
# ax.axhline(0, linewidth=0.2, color="black")
# ax.legend(loc=2)
# plt.show()

# figure = plt.figure(figsize=(20, 10))
# ax = figure.add_subplot(111)
# for key in lab_pca.keys():
#     ax.plot(stc.times, lab_pca[key], label=key)
# ax.axvline(0, linewidth=0.2, color="black")
# ax.axvline(1.6, linewidth=0.2, color="black")
# ax.axhline(0, linewidth=0.2, color="black")
# ax.legend(loc=2)
# plt.show()

