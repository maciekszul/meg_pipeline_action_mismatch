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
        try:
            raw_start = raw.times[events[0][0]] - 5
            raw_end = raw.times[events[-1][0]] + 5
            raw = raw.crop(tmin=raw_start, tmax=raw_end)
        except:
            raw_start = raw.times[events[0][0]] - 5
            raw_end = raw.times[events[-1][0]] + 1
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
            ref_meg=True,
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

        raw.save(raw_80_out, overwrite=True)
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
        max_iter = 10000

        ica = ICA(
            n_components=n_components, 
            method=method,
            max_iter=max_iter
        )

        ica.fit(
            raw,
        )

        ica.save(
            ica_out
        )
        print(ica_out)

        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        print("step 2 done:", time_string)


if parameters["step_3"]:
    components_file_path = op.join(
        subject_meg,
        "rejected-components.json"
    )

    with open(components_file_path) as data:
        components_rej = json.load(data)
    
    for raw_file in components_rej.keys():
        # output paths
        epochs_TF = op.join(
            subject_meg,
            "epochs-TF-{}-epo.fif".format(raw_file[3:6])
        )
        epochs_TD = op.join(
            subject_meg,
            "epochs-TD-{}-epo.fif".format(raw_file[3:6])
        )

        # read files
        raw = mne.io.read_raw_fif(
            op.join(subject_meg, raw_file),
            preload=True
        )

        ica_path = files.get_files(
            subject_meg,
            "{}".format(raw_file[:6]),
            "-ica.fif",
            wp=True
        )[2][0]

        events_path = files.get_files(
            subject_meg,
            "events-{}".format(raw_file[3:6]),
            "-eve.fif",
            wp=True
        )[2][0]
        
        events = mne.read_events(
            events_path
        )

        ica = mne.preprocessing.read_ica(ica_path)

        raw = ica.apply(
            raw,
            exclude=components_rej[raw_file]
        )

        # calculating the mid-phase cutout

        tr_start = np.where((events[:,2] < 50))[0]
        tr_end = np.where((events[:,2] > 50) & (events[:,2] < 80))[0]

        tr_onsets, tr_ends = events[tr_start][:,0], events[tr_end][:,0]

        tr_durations = tr_ends - tr_onsets
        
        tr_cutouts = np.int_(tr_durations - 1.5*250)


        # TF epochs
        epochs = mne.Epochs(
            raw,
            events=events,
            baseline=None,
            preload=True,
            event_id=[30, 40],
            tmin=-0.5,
            tmax=np.max(tr_durations/250 + 1.1 + 2.5),
            detrend=1
        )

        epochs_array = []
        for ix, epo in enumerate(list(epochs.iter_evoked())):
            data = epo.data
            del_ints = np.arange(500, 500 + tr_cutouts[ix])
            data = np.delete(data, del_ints, axis=1)
            data = data[:,:775]
            epochs_array.append(data)
        
        epochs_array = np.array(epochs_array)

        epochs = mne.EpochsArray(
            epochs_array,
            epochs.info,
            events=events[tr_start[:epochs_array.shape[0]]],
            tmin=-0.5,
            baseline=None
        )

        epochs.save(epochs_TF)

        del epochs

        # TD epochs

        raw = raw.filter(
            None,
            30,
            method="fir",
            phase="minimum",
            n_jobs=-1
        )

        epochs = mne.Epochs(
            raw,
            events=events,
            baseline=None,
            preload=True,
            event_id=[30, 40],
            tmin=-0.5,
            tmax=np.max(tr_durations/250 + 1.1 + 2.5),
            detrend=1
        )

        epochs_array = []
        for ix, epo in enumerate(list(epochs.iter_evoked())):
            data = epo.data
            del_ints = np.arange(500, 500 + tr_cutouts[ix])
            data = np.delete(data, del_ints, axis=1)
            data = data[:,:775]
            epochs_array.append(data)
        
        epochs_array = np.array(epochs_array)

        epochs = mne.EpochsArray(
            epochs_array,
            epochs.info,
            events=events[tr_start[:epochs_array.shape[0]]],
            tmin=-0.5,
            baseline=None
        )

        epochs.save(epochs_TD)

        del epochs

        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        print("step 3 done:", time_string)


if parameters["step_4"]:
    # setup a source space
    src = mne.setup_source_space(
        subject=subject, 
        subjects_dir=subjects_dir, 
        spacing="ico5", 
        add_dist=False,
        n_jobs=2
    )

    src_path = op.join(
        subject_meg,
        "{}-src.fif".format(subject)
    )

    mne.write_source_spaces(src_path, src, overwrite=True)

    conductivity = (0.3, )
    
    model = mne.make_bem_model(
        subject=subject,
        ico=5,
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )

    bem = mne.make_bem_solution(model)

    bem_path = op.join(
        subject_meg,
        "{}-bem.fif".format(subject)
    )

    mne.write_bem_solution(bem_path, bem)

    del src
    del bem
    
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("step 4 done:", time_string)


if parameters["step_5"]:
    # read files
    src_path = op.join(
        subject_meg,
        "{}-src.fif".format(subject)
    )
    src = mne.read_source_spaces(src_path)

    bem_path = op.join(
        subject_meg,
        "{}-bem.fif".format(subject)
    )
    bem = mne.read_bem_solution(bem_path)

    trans_files = files.get_files(
        subject_meg,
        "",
        "-trans.fif"
    )[2]
    trans_files.sort()

    raws_files = files.get_files(
        subject_meg,
        "80",
        "-raw.fif"
    )[2]
    raws_files.sort()

    for ix, (raw_path, trans_path) in enumerate(zip(raws_files, trans_files)):
        raw = mne.io.read_raw_fif(raw_path)
        fwd = mne.make_forward_solution(
            raw.info,
            trans=trans_path,
            src=src,
            bem=bem,
            meg=True,
            mindist=0.5,
            n_jobs=-1
        )

        fwd_path = op.join(
            subject_meg,
            "{}-fwd.fif".format(str(ix).zfill(3))
        )
        mne.write_forward_solution(
            fwd_path,
            fwd,
            overwrite=True
        )
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("step 5 done:", time_string)


if parameters["step_6"]:
    src_path = op.join(
        subject_meg,
        "{}-src.fif".format(subject)
    )

    src = mne.read_source_spaces(src_path)

    mne.compute_source_morph(
        src,
        subject_from=subject,
        subject_to="fsaverage",
        subjects_dir=subjects_dir,
        spacing=5,
        smooth=None
    )


if parameters["step_7"]:

    if parameters["cov_mx_pre"] == "epochs-TD":

        epochs_files = files.get_files(
            subject_meg,
            parameters["cov_mx_pre"],
            "-epo.fif",
            wp=True
        )[2]
        epochs_files.sort()
        for ix, epo_path in enumerate(epochs_files):
            epochs = mne.read_epochs(
                epo_path,
                preload=True
            )
            epochs = epochs.apply_baseline((-0.1, 0.0))
            epochs = epochs.apply_baseline((1.6, 2.6))

            cov_mx = mne.compute_covariance(
                epochs,
                method="auto",
                cv=5,
                scalings=dict(mag=1e13, grad=1e15, eeg=1e6), # because the data is from gradiometers see the https://mne-tools.github.io/0.17/generated/mne.compute_covariance.html#mne.compute_covariance
                n_jobs=-1,
                rank=None
            )
            cov_mx_path = op.join(
                subject_meg,
                "{0}-{1}-cov.fif".format(parameters["cov_mx_pre"], str(ix).zfill(3))
            )
            cov_mx.save(cov_mx_path)

            named_tuple = time.localtime() # get struct_time
            time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
            print("step 7 done:", time_string)