import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)

import mne
from mne.preprocessing import ICA
import os.path as op
import json
import argparse
from tools import files
import numpy as np
import pandas as pd

json_file = "pipeline_params.json"

# argparse input
des = "pipeline script"
parser = argparse.ArgumentParser(description=des)
parser.add_argument(
    "-f", 
    type=str, 
    nargs=1,
    default=json_file,
    help="JSON file with pipeline parameters"
)

parser.add_argument(
    "-n", 
    type=int, 
    help="id list index"
)

args = parser.parse_args()
params = vars(args)
json_file = params["f"][0]
subj_index = params["n"]

# read the pipeline params
with open(json_file) as pipeline_file:
    pipeline_params = json.load(pipeline_file)

# paths
data_path = pipeline_params["data_path"]
fs_path = pipeline_params["fs_output"]

subjs = files.get_folders_files(fs_path, wp=False)[0]
subjs.sort()
subjs = [i for i in subjs if "fsaverage" not in i]
subj = subjs[subj_index]

meg_subj_path = op.join(data_path,"MEG", subj)
beh_subj_path = op.join(data_path,"BEH", subj)

verb=False

print(subj)
print(meg_subj_path)
print(beh_subj_path)

if pipeline_params["downsample_convert_filter"]:
    raw_ds = files.get_folders_files(
        meg_subj_path,
        wp=True
    )[0]
    raw_ds = [i for i in raw_ds if ".ds" in i]

    for ix, raw_path in enumerate(raw_ds):
        raw = mne.io.read_raw_ctf(
            raw_path,
            preload=True,
            verbose=False
        )

        picks_meg = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=True, 
            eog=True, 
            ecg=False, 
            ref_meg=False
        )

        events = mne.find_events(
            raw,
            stim_channel="UPPT001",
            min_duration=0.003
        )

        raw, events = raw.copy().resample(
            pipeline_params["downsample_to"], 
            npad="auto", 
            events=events,
            n_jobs=-1,
        )
        print(subj, ix, "resampled")
        raw = raw.filter(
            0.1,
            80,
            picks=picks_meg,
            n_jobs=-1,
            method="fir",
            phase="minimum"
        )
        print(subj, ix, "filtered")
        raw_out_path = op.join(
            meg_subj_path,
            "raw-{}-raw.fif".format(str(ix).zfill(3))
        )
        events_out_path = op.join(
            meg_subj_path,
            "{}-eve.fif".format(str(ix).zfill(3))
        )

        raw.save(raw_out_path, overwrite=True)
        mne.write_events(events_out_path, events)
        print(subj, ix, "saved")


if pipeline_params["epochs"]:
    raw_files = files.get_files(
        meg_subj_path,
        "raw",
        "-raw.fif",
        wp=True
    )[2]
    raw_files.sort()

    eve_files = files.get_files(
        meg_subj_path,
        "",
        "-eve.fif",
        wp=True
    )[2]
    eve_files.sort()

    beh_files = files.get_files(
        beh_subj_path,
        "ses",
        ".pkl",
        wp=True
    )[2]
    beh_files.sort()

    all_files = zip(raw_files, eve_files, beh_files)
    samp =  pipeline_params["downsample_to"]
    for raw_file, event_file, beh_file in all_files:

        raw = mne.io.read_raw_fif(
            raw_file,
            preload=True,
            verbose=verb
        )

        raw = raw.filter(1, 30)

        picks_meg = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=True, 
            eog=True, 
            ecg=False, 
            ref_meg=False
        )

        beh_data = pd.read_pickle(beh_file)

        events_rot = mne.read_events(
            event_file,
            include=[30, 40]
        )

        events_obs = mne.read_events(
            event_file,
            include=[60, 70]
        )
        beh_data = beh_data.loc[(beh_data.obs_dir_mod != 0)]

        resp_onset = beh_data.action_onset.values / samp

        obs_onset = (events_obs[:,0] - events_rot[:,0]) / samp

        tmin, tmax = (-0.5, max(obs_onset)+1)
        baseline = (tmin, 0.0)
        big_epochs = mne.Epochs(
            raw,
            events_rot,
            picks=picks_meg,
            tmin=tmin,
            tmax=tmax,
            detrend=1,
            baseline=baseline
        )
        big_epochs.apply_baseline(baseline)

        rotation = []
        observation = []
        obs_trig = []
        for ix, evo in enumerate(big_epochs.iter_evoked()):

            obs_ons = obs_onset[ix]
            obs = evo.copy()
            obs.crop(obs_ons-0.1, obs_ons+1)
            observation.append(obs)
        
        obs_all = np.array([i.data for i in observation])
        obs_all = np.array(obs_all)  

        obs_file_out = op.join(
            meg_subj_path,
            "obs-{}-epo.fif".format(op.split(raw_file)[1].split("-")[1])
        )

        try:
            obs_epo = mne.EpochsArray(
                obs_all,
                big_epochs.info,
                events=events_rot, # investigate where the last epoch goes during iter_evoked() IMPORTANT!
                tmin=-0.1,
                baseline=(-0.1, 0.0)
            )
        except:
            obs_epo = mne.EpochsArray(
                obs_all,
                big_epochs.info,
                events=events_rot[:obs_all.shape[0]], # investigate where the last epoch goes during iter_evoked() IMPORTANT!
                tmin=-0.1,
                baseline=(-0.1, 0.0)
            )

        obs_epo.save(obs_file_out)

if pipeline_params["ica_epochs"]:
    epo_files = files.get_files(
        meg_subj_path,
        "obs-",
        "-epo.fif",
        wp=True
    )[2]

    epo_files.sort()
    
    for ix, epo in enumerate(epo_files):
        ica_out_path = op.join(
            meg_subj_path,
            "obs-{0}-ica.fif".format(str(ix).zfill(3))
        )

        epoch = mne.read_epochs(
            epo,
            preload=True,
            verbose=verb
        )

        n_components = 50
        method = "fastica"
        reject = dict(mag=4e-12)
        max_iter = 10000

        ica = ICA(
            n_components=n_components, 
            method=method,
            max_iter=max_iter
        )

        ica.fit(
            epoch, 
            reject=reject,
            verbose=verb
        )

        ica.save(ica_out_path)
        print(subj, ix, "saved")

if pipeline_params["apply_ICA"]:
    ica_json = files.get_files(
        meg_subj_path,
        "",
        "ica-rej.json"
    )[2][0]

    raw_files = files.get_files(
        meg_subj_path,
        "obs-",
        "-epo.fif",
        wp=False
    )[2]

    comp_ICA_json_path = op.join(
        meg_subj_path,
        "obs-{}-ica-rej.json".format(str(subj).zfill(3))
    )

    ica_files = files.get_files(
        meg_subj_path,
        "obs-",
        "-ica.fif",
        wp=False
    )[2]
    
    with open(ica_json) as data:
        components_rej = json.load(data)

    for k in components_rej.keys():
        raw_path = op.join(
            meg_subj_path,
            files.items_cont_str(raw_files, k, sort=True)[0]
        )
        ica_path = op.join(
            meg_subj_path,
            files.items_cont_str(ica_files, k, sort=True)[0]
        )
        
        obs_path = op.join(
            meg_subj_path, 
            "ica-obs-{0}-epo.fif".format(k)
        )
        raw = mne.read_epochs(
            raw_path,
            preload=True,
            verbose=verb
        )

        ica = mne.preprocessing.read_ica(ica_path)
        raw_ica = ica.apply(
            raw,
            exclude=components_rej[k]
        )
        
        raw_ica.save(
            obs_path,
            fmt="single",
            split_size="2GB"
        )

        print(raw_path)

if pipeline_params["fwd_solution"]:
    src = mne.setup_source_space(
        subject=subj, 
        subjects_dir=fs_path, 
        spacing="ico5", 
        add_dist=False
    )

    src_file_out = op.join(
        meg_subj_path,
        "{}-src.fif".format(subj)
    )

    mne.write_source_spaces(src_file_out, src)

    conductivity = (0.3,)
    model = mne.make_bem_model(
        subject=subj,
        ico=5,
        conductivity=conductivity,
        subjects_dir=fs_path
    )

    bem = mne.make_bem_solution(model)

    bem_file_out = op.join(
        meg_subj_path,
        "{}-bem.fif".format(subj)
    )
    mne.write_bem_solution(bem_file_out, bem)

    raw_files = files.get_files(
        meg_subj_path,
        "raw",
        "-raw.fif"
    )[2]
    raw_files.sort()

    epo_files = files.get_files(
        meg_subj_path,
        "all",
        "-epo.fif"
    )[2]
    epo_files.sort()

    trans_file = op.join(
        meg_subj_path,
        "{}-trans.fif".format(subj)
    )

    all_files = zip(raw_files, epo_files)
    for raw_file, epo_file in all_files:
        file_id = op.split(raw_file)[1].split("-")[1]

        fwd_out = op.join(
            meg_subj_path,
            "fwd-{}-fwd.fif".format(file_id)
        )

        fwd = mne.make_forward_solution(
            raw_file,
            trans=trans_file,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=-1
        )
        
        mne.write_forward_solution(
            fwd_out, 
            fwd, 
            verbose=verb
        )

        print(fwd_out)

if pipeline_params["cov_matrix"]:
    raw_files = files.get_files(
        meg_subj_path,
        "raw",
        "-raw.fif"
    )[2]
    raw_files.sort()
    for raw_file in raw_files:
        file_id = op.split(raw_file)[1].split("-")[1]

        cov_mx_out = op.join(
            meg_subj_path,
            "mx-{}-cov.fif".format(file_id)
        )

        raw = mne.io.read_raw_fif(
            raw_file, 
            preload=True,
            verbose=verb
        )

        picks = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            stim=False, 
            eog=False, 
            ref_meg="auto", 
            exclude="bads"
        )

        noise_cov = mne.compute_raw_covariance(
            raw, 
            method="auto", 
            rank=None,
            picks=picks,
            n_jobs=-1
        )

        noise_cov.save(
            cov_mx_out
        )

        print(cov_mx_out)

if pipeline_params["inv_operator"]:

    epo_files = files.get_files(
        meg_subj_path,
        "all",
        "-epo.fif"
    )[2]
    epo_files.sort()
    cov_files = files.get_files(
        meg_subj_path,
        "mx",
        "-cov.fif"
    )[2]
    cov_files.sort()
    fwd_files = files.get_files(
        meg_subj_path,
        "fwd",
        "-fwd.fif"
    )[2]
    fwd_files.sort()

    all_files = zip(epo_files, cov_files, fwd_files)

    for epo_path, cov_path, fwd_path in all_files:
        file_id = op.split(epo_path)[1].split("-")[1]

        inv_out = op.join(
            meg_subj_path,
            "inv-{}-inv.fif".format(file_id)
        )

        fwd = mne.read_forward_solution(fwd_path)

        cov = mne.read_cov(cov_path)

        epochs = mne.read_epochs(epo_path)

        inv = mne.minimum_norm.make_inverse_operator(
            epochs.info,
            fwd,
            cov,
            loose=0.2,
            depth=0.8
        )

        mne.minimum_norm.write_inverse_operator(
            inv_out,
            inv
        )

        print(inv_out)


if pipeline_params["compute_inverse"][0]:
    method_dict = {
        "dSPM": (8, 12, 15),
        "sLORETA": (3, 5, 7),
        "eLORETA": (0.75, 1.25, 1.75)
    }

    method = pipeline_params["compute_inverse"][1]
    snr = 3.
    lambda2 = 1. / snr ** 2
    lims = method_dict[method]

    epo_files = files.get_files(
        meg_subj_path,
        "all",
        "-epo.fif"
    )[2]
    epo_files.sort()

    inv_files = files.get_files(
        meg_subj_path,
        "inv",
        "-inv.fif"
    )[2]
    inv_files.sort()

    all_files = zip(epo_files, inv_files)

    for epo_path, inv_path in all_files:
        file_id = op.split(epo_path)[1].split("-")[1]

        stc_out = op.join(
            meg_subj_path,
            "stc-{}"
        )

        epo = mne.read_epochs(
            epo_path,
            verbose=verb,
            preload=True
        )

        epo = epo.average()

        inv = mne.minimum_norm.read_inverse_operator(
            inv_path,
            verbose=verb
        )

        stc = mne.minimum_norm.apply_inverse(
            epo,
            inv,
            lambda2,
            method=method,
            pick_ori=None,
            verbose=True
        )

        # stc.save(stc_out)


named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("end:", time_string)