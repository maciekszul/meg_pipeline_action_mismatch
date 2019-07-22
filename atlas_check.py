import mne
from surfer import Brain
import numpy as np
import pandas as pd
from operator import itemgetter

lh_colortab = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/lh.hcp-mmp-b_colortab.txt"
rh_colortab = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/rh.hcp-mmp-b_colortab.txt"
lh_annotation = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/lh.hcp-mmp-b.annot"
rh_annotation = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/rh.hcp-mmp-b.annot"
lut = "/cubric/data/c1557187/atlas_data/hcp-mmp-b/LUT_hcp-mmp-b.txt"

brain = Brain(
    subject_id="fsaverage",
    subjects_dir="/cubric/scratch/c1557187/MRI_337/FS_OUTPUT",
    hemi="both",
    surf="inflated"
)

