# meg_pipeline_action_mismatch


Step 1:
 - read in *.ds raw data structure
 - crop raw file
 - find events
 - filter 0.1~ and ~80 Hz applied
 - downsampling raw + events
 - save files


Step 2:
 - cut out the joystick part,
 - fit ICA
 - save solution

MANUALLY ANNOTATE COMPONENTS TO REJECT

Step 3:
 - apply the rejection
 - separately filter the raw ~30Hz (for T domain)
 - epoch for T-F and T domain
 - save both

Step 3.5:
 - apply the rejection
 - separately filter the raw ~30Hz (for T domain)
 - epochs aligned to the onset of the trial
 - epoch for T-F and T domain
 - save both

MANUALLY FIT THE HEADSHAPE

Step 4:
 - prepare for forward solution
 - anatomy only
 - setup a source space
 - calculate bem solution

Step 5:
 - calculate forward solution

Step 6:
 - compute a spherical morphing from subject space to fsaverage

Step 7:
 - pick the epochs
 - compute teh appropriate covariance matrix

Step 8:
 - pick the epochs
 - compute the inverse model