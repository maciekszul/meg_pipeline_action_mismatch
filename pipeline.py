import mne
from mne.preprocessing import ICA
import os.path as op
import json
import argparse
from tools import files

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
json_file = params["f"]
subj_index = params["n"]

# read the pipeline params
with open(json_file) as pipeline_file:
    pipeline_params = json.load(pipeline_file)

# paths
data_path = pipeline_params["data_path"]
fs_path = op.join(data_path, "MRI")

subjs = files.get_folders_files(fs_path, wp=False)[0]
subjs.sort()
subjs = [i for i in subjs if "fsaverage" not in i]
subj = subjs[subj_index]

meg_subj_path = op.join(data_path,"MEG", subj)
beh_subj_path = op.join(data_path,"BEH", subj)

verb=False

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
            eeg=False, 
            eog=False, 
            ecg=False, 
            ref_meg=False
        )

        events = mne.find_events(
            raw,
            stim_channel="UPPT001"
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

        ica_out_path = op.join(
            meg_subj_path,
            "{}-ica.fif".format(str(ix).zfill(3))
        )

        n_components = 50
        method = "extended-infomax"
        reject = dict(mag=4e-12)

        ica = ICA(
            n_components=n_components, 
            method=method
        )

        ica.fit(
            raw, 
            picks=picks_meg,
            reject=reject,
            verbose=verb
        )
        print(subj, ix, "ICA_fit")
        raw.save(raw_out_path, overwrite=True)
        mne.write_events(events_out_path, events)
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
        "raw",
        "-raw.fif",
        wp=False
    )[2]

    comp_ICA_json_path = op.join(
        meg_subj_path,
        "{}-ica-rej.json".format(str(subj).zfill(3))
    )

    ica_files = files.get_files(
        meg_subj_path,
        "",
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
        
        raw = mne.io.read_raw_fif(
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
            raw_path,
            fmt="single",
            split_size="2GB",
            overwrite=True
        )

        print(raw_path)