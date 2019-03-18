import matplotlib
matplotlib.use("Qt5Agg")
import mne.gui
import os.path as op
import json
import argparse
import tools

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


meg_path = op.join(data_path, "MEG")