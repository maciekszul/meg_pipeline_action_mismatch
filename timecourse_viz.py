from tools import files
import numpy as np
import matplotlib.pylab as plt
import os.path as op
import sys
from matplotlib import gridspec

try:
    file_index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    side = int(sys.argv[2])
except:
    print("incorrect arguments")
    sys.exit()

main_path = "/cubric/scratch/c1557187/act_mis"

output = op.join(
    main_path,
    "RESULTS",
    "TD_SOURCE_SPACE_AVG"
)

data_files = files.get_files(
    output,
    "",
    ".npy"
)[2]

data = np.load(data_files[file_index]).item()

print(data_files[file_index])

times = np.linspace(-0.5, 2.596, num=775)


left = [i for i in data.keys()if "L_" in i]
right = [i for i in data.keys()if "R_" in i]

pick = [left, right][side]

columns = 4
rows = np.int(len(pick)/columns) + 1

gs = gridspec.GridSpec(rows, columns, hspace=1)
figure = plt.figure(figsize=(20, 10))
for ix, key in enumerate(pick):
    signal = data[key].reshape(-1)
    ax = figure.add_subplot(gs[ix])
    ax.plot(times, signal/10, linewidth=0.5)
    ax.axvline(0, linewidth=0.2, color="black")
    ax.axvline(1.6, linewidth=0.2, color="black")
    ax.axhline(0, linewidth=0.2, color="black")
    ax.title.set_text(key)
# plt.tight_layout()
plt.show()