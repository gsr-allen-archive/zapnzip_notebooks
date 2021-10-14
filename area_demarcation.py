# %% codecell
import os
from glob import glob
import json

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, patches
# import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm
import pandarallel
from IPython.utils.capture import capture_output
with capture_output():
    tqdm.pandas()
    pandarallel.pandarallel.initialize(progress_bar=True)

from tbd_eeg.data_analysis.eegutils import EEGexp
from tbd_eeg.data_analysis.Utilities.utilities import get_stim_events, find_nearest_ind

%matplotlib inline
# from ipympl.backend_nbagg import Canvas
# Canvas.header_visible.default_value = False

# %% codecell
# accessing the Google sheet with experiment metadata in python
# setting up the permissions:
# 1. install gspread (pip install gspread / conda install gspread)
# 2. copy the service_account.json file to '~/.config/gspread/service_account.json'
# 3. run the following:
import gspread
_gc = gspread.service_account() # need a key file to access the account (step 2)
_sh = _gc.open('Zap_Zip-log_exp') # open the spreadsheet
_df = pd.DataFrame(_sh.sheet1.get()) # load the first worksheet
gmetadata = _df.T.set_index(0).T # put it in a nicely formatted dataframe
gmetadata

# %% codecell
rec_folder = 'tiny-blue-dot/zap-n-zip/EEG_exp/mouse543396/estim_vis1_2020-09-18_12-04-46/experiment1/recording1/'
exp = EEGexp(rec_folder, preprocess=False, make_stim_csv=False)

#%% codecell
# Let's print some meta data
print('Mouse: {}'.format(exp.mouse))
print('Experiment date: {}'.format(exp.date))
print('What data is in here?')
print(exp.experiment_data)

#%% codecell
stim_log = pd.read_csv(exp.stimulus_log_file)
stim_log.head()

#%% codecell
fig, ax = plt.subplots(figsize=(3.5, 3), constrained_layout=True)

ax.scatter(exp.EEG_channel_coordinates['ML'], exp.EEG_channel_coordinates['AP'], s=300, color='orange')
ax.scatter(0, 0, marker='P', color='red')
ax.axis('equal')

for ind in range(len(exp.EEG_channel_coordinates)):
    ax.annotate(str(ind),  xy=(exp.EEG_channel_coordinates['ML'].iloc[ind], exp.EEG_channel_coordinates['AP'].iloc[ind]), ha='center', va='center', color="k")

ax.set_xlabel("ML axis (mm)\nmouse's left <--> right")
ax.set_ylabel('AP axis (mm)')
ax.set_title('NeuroNexus numbering')
fig.show()

#%% codecell
eeg_data = exp.load_eegdata(frequency=2500, return_type='pd')
plot_ch = 27 # choose which electrode to plot (zero-indexed, ch 30:31 do not exist)

fig, ax = plt.subplots(figsize=(7, 2.4), tight_layout=True)
eeg_data[plot_ch].plot(ax=ax)

# plot cosmetics
ax.set_xlabel('Time (s)')
ax.set_ylabel('Raw signal (uV)')
ax.set_title('EEG channel %d' % plot_ch);
fig.show()

#%% codecell
probe = 'probeC'
lfp = np.memmap(exp.ephys_params[probe]['lfp_continuous'], dtype='int16', mode='r')
lfp = np.reshape(lfp, (int(lfp.size/exp.ephys_params[probe]['num_chs']), exp.ephys_params[probe]['num_chs']))
samp_rate = exp.ephys_params[probe]['lfp_sample_rate']
timestamps = np.load(exp.ephys_params[probe]['lfp_timestamps'])

t_lim_s = np.array([60, 100])*samp_rate
sampled_lfp = lfp[slice(*t_lim_s), :]

#%% codecell
v = np.quantile(sampled_lfp, q=[0.05, 0.95])
f, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
ax.imshow(sampled_lfp, aspect='auto', cmap=cm.bwr, vmin=v[0], vmax=v[1])
f.show()
