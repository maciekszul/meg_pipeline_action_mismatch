import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)
import mne
from mne.preprocessing import ICA
import os.path as op
import json
from tools import files
import numpy as np
import pandas as pd
import sys


# parsing command line arguments
try:
    subj_index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[2]
    print(json_file)
except:
    json_file = "pipeline.json"
    print(json_file)

# open json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

# prepare paths
data_path = parameters["path"]
subjects_dir = parameters["freesurfer"]

subjects = files.get_folders_files(subjects_dir, wp=False)[0]
subjects.sort()
subjects = [i for i in subjects if "fsaverage" not in i]
subject = subjects[subj_index]

print("subject:", subject)

subject_raw = op.join(
    data_path,
    "MEG",
    subject
)

subject_meg = op.join(
    data_path,
    "MEG",
    subject,
    parameters["folder"]
)

files.make_folder(subject_meg)

subject_beh = {
    data_path,
    "BEH",
    subject
}

if parameters["step_1"]:

    #read raw ds
    ds = files.get_folders_files(
        subject_raw,
        wp=True
    )[0]

    ds_list = [i for i in ds if ".ds" in i]

    for ix, ds in enumerate(ds_list):

        # output paths
        raw_80_out = op.join(
            subject_meg,
            "80-{0}-raw.fif".format(str(ix).zfill(3))
        )

        eve_out = op.join(
            subject_meg,
            "events-{0}-eve.fif".format(str(ix).zfill(3))
        )

        # processing

        raw = mne.io.read_raw_ctf(ds, preload=True)

        events = mne.find_events(
            raw,
            stim_channel="UPPT001",
            min_duration=0.003
        )

        # cropping raw file 5s relative to first and last event
        raw_start = raw.times[events[0][0]] - 5
        raw_end = raw.times[events[-1][0]] + 5
        raw = raw.crop(tmin=raw_start, tmax=raw_end)

        events = mne.find_events(
            raw,
            stim_channel="UPPT001",
            min_duration=0.003
        )

        set_ch = {'EEG057-3305':'eog', 'EEG058-3305': 'eog'}
        raw.set_channel_types(set_ch)

        raw = raw.pick_types(
            meg=True,
            ref_meg=False,
            eog=True,
            eeg=False
        )

        raw = raw.filter(
            0.1,
            None,
            method="fir",
            phase="minimum",
            n_jobs=-1
        )
        
        raw = raw.filter(
            None,
            80,
            method="fir",
            phase="minimum",
            n_jobs=-1
        )

        

        raw, events = raw.copy().resample(
            250, 
            npad="auto", 
            events=events,
            n_jobs=-1,
        )

        raw.save(raw_80_out)
        print(raw_80_out)

        mne.write_events(eve_out, events)

        # deleting  heavy objects
        del raw

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("step 1 done:", time_string)

if parameters["step_2"]:
    raws_files = files.get_files(
        subject_meg,
        "80",
        "-raw.fif",
        wp=True
    )[2]
    raws_files.sort()

    events_files = files.get_files(
        subject_meg,
        "events",
        "-eve.fif",
        wp=True
    )[2]
    events_files.sort()

    for ix, (raw_path, event_path) in enumerate(zip(raws_files, events_files)):
        
        # output paths
        ica_out = op.join(
            subject_meg,
            "80-{0}-ica.fif".format(str(ix).zfill(3))
        )

        events = mne.read_events(event_path)
        raw = mne.io.read_raw_fif(raw_path, preload=True)

        # find indices (ints) of the rows that will be deleted from raw timecourse
        start = np.where((events[:,2] < 50))[0]
        end = np.where((events[:,2] > 50) & (events[:,2] < 80))[0]
        del_ranges = list(zip(events[start][:,0], events[end][:,0]))
        ints = np.concatenate([np.arange(i[0], i[1]) for i in del_ranges])

        # operations on raw array
        data = raw.get_data()
        data = np.delete(data, ints, axis=1)

        # recreating the raw object
        raw = mne.io.RawArray(data, raw.info)

        # ICA
        n_components = 50
        method = "fastica"
        reject = dict(mag=4000e-13)
        max_iter = 10000

        ica = ICA(
            n_components=n_components, 
            method=method,
            max_iter=max_iter
        )

        ica.fit(
            raw,
            reject=reject
        )

        ica.save(
            ica_out
        )
        print(ica_out)


named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("step 2 done:", time_string)