import mne


inv = mne.minimum_norm.read_inverse_operator("/cubric/scratch/c1557187/act_mis/MEG/0001/new_v1/epochs-TF-001-inv.fif")
epo = mne.read_epochs("/cubric/scratch/c1557187/act_mis/MEG/0001/new_v1/epochs-TF-001-epo.fif", preload=True)

subjects_dir = "/cubric/scratch/c1557187/MRI_337/FS_OUTPUT"
subject = "0001"

source = mne.minimum_norm.apply_inverse(
    epo[100].average(),
    inv,
    lambda2= 1/3**2,
    method="dSPM"
)

source.plot(subjects_dir="/cubric/scratch/c1557187/MRI_337/FS_OUTPUT", subject="0001", time_viewer=True) 
