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