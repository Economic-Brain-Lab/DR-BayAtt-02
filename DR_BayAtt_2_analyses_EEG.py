'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2020-09-16
-----
Last Modified: 2020-09-23
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2020 Dragan Rangelov, The University of Queensland
'''
# DONE: stimulus-locked motion decoding (forward-encoding analyses)
# DONE: stimulus-locked decision decoding (effect-matched spatial filtering)
# NOTE: response-locked epochs unnecessary, epoching locked to trial onset
# DONE: response-locked motion decoding
# DONE: response-locked decision decoding
# NOTE: Mass-univariate analyses: 
# 1. ratio b/w hEOG/vEOG
# 2. decreasing visual noise in figure - average across electrode meridian
#===============================================================================
# %% setting paths
#===============================================================================
rootPath = '/dune/Experiments/'
# rootPath = '/Users/uqdrange/Experiments/'
scriptPath = rootPath + '00-Scripts/'
expPath = rootPath + 'DR-BayAtt-2/'
bhvPath = expPath + 'Data/'
dataPath = expPath + 'EEG/raw/'
montagePath = expPath + 'EEG/'
exportPath = expPath + 'Export/'
savePath = '/dune/Scratch/DR-BayAtt-2/'
# savePath = '/Users/Shared/Scratch/DR-BayAtt-2/'
#===============================================================================
# %% set plotting
#===============================================================================
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sbn
sbn.set()
sbn.set_style('ticks')
#===============================================================================
# %% importing libraries
#===============================================================================
# %matplotlib qt5
import sys
sys.path.insert(0, scriptPath)
sys.path.insert(0, expPath + '/Scripts')
import logging
import os
import numpy as np
import mne
import scipy
import pandas as pd
import EEG_functions as eegfun
import mmodel as mm
import plot_sensors_custom as ps
import pickle
import gzip
# os.environ['R_HOME'] = '/home/ubuntu/anaconda3/envs/dev/lib/R'
# # os.environ['R_HOME'] = '/Users/uqdrange/anaconda2/envs/dev/lib/R'
# %load_ext rpy2.ipython
from collections import OrderedDict
from distMahalanobis import compute_MahDist
from statsmodels import distributions as dist
from statsmodels.stats import multitest
from scipy import stats
import h5py
from mne.decoding import EMS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#===============================================================================
# %% reading in raw data and running pre-processing routine for ERPs
#===============================================================================
dsets = [dfile[:-8] for dfile in os.listdir(dataPath) if dfile not in ['.DS_Store','._.DS_Store']]
dropsets = ['s09'] # dropping dsets which were not recorded correctly
dsets = sorted(filter(lambda x: x not in dropsets, dsets))
# #==============================================================================
# # setting electrodes to remove, rename, and recompute
# #==============================================================================
# # %%
# dropElectrodes = [u'EXG1',u'EXG2',u'EXG7',u'EXG8',u'Nz',u'SNz',u'M1',u'M2'] # dropping electrodes that were nor recorded
# eogElectrodes = [u'LH',u'RH',u'LV',u'UV']
# renameElectrodes = {u'EXG3':u'LH',u'EXG4':u'RH',u'EXG5':u'LV',u'EXG6':u'UV'}
# #===============================================================================
# # EEG pre-processing
# #===============================================================================
# for dset in dsets:
#     dataEEG = {}
#     badChannels = {}
#     badEpochs = {}
#     badICA = {}
#
#     # reading in raw data
#     logging.info('reading in raw data for {}'.format(dset))
#     crop = [] # this list should contain times points (in seconds) which should be cropped from the eeg files
#     if dset == 's31': crop = [276.59,280]
#     dataEEG[dset] = {'raw': eegfun.readRawData(dset, dataPath, savePath, montagePath, dropElectrodes, eogElectrodes, renameElectrodes, crop = crop)}
#     # REVIEW: s03, s04 have trigger 103 - which is impossible
#     # REVIEW: some participants have trial number triggers > 49 (56-63, inclusive)
#     # NOTE: s31 recording was interrupted, missing trials 192-255 inclusive
#     dataEEG[dset]['events'] = mne.find_events(dataEEG[dset]['raw'], shortest_event = 1)
#
#     # re-referencing
#     logging.info('re-referencing for {}'.format(dset))
#     dataEEG[dset]['avRef'] = eegfun.reReferenceData(dataEEG[dset]['raw'])
#
#     # low-pass filtering eeg electrodes
#     logging.info('low-pass filtering eeg electrodes for {}'.format(dset))
#     dataEEG[dset]['filtered'] = dataEEG[dset]['avRef'].copy()
#     picks = mne.pick_types(dataEEG[dset]['filtered'].info, meg=False, eeg=True)
#     dataEEG[dset]['filtered'].filter(l_freq=None, h_freq=99, picks=picks)
#     dataEEG[dset]['filtered'].notch_filter(50, picks)
#
#     # band-pass filtering eog electrodes
#     logging.info('band-pass filtering eog electrodes for {}'.format(dset))
#     picks = mne.pick_types(dataEEG[dset]['filtered'].info, meg=False, eog=True)
#     dataEEG[dset]['filtered'].filter(l_freq=1, h_freq=10, picks=picks)
#
#     # interpolating bad channels
#     logging.info('interpolating bad channels for {}'.format(dset))
#     eegfun.findBadChannels(dataEEG[dset]['filtered'])
#     badChannels[dset] = dataEEG[dset]['filtered'].info['bads']
#     dataEEG[dset]['interpolated'] = eegfun.fixBadChannels(dataEEG[dset]['filtered'])
#
#     # re-referencing channels
#     logging.info('re-referencing channels for {}'.format(dset))
#     dataEEG[dset]['interpolated'] = eegfun.reReferenceData(dataEEG[dset]['interpolated'])
#
#     # epoching with regard to buffer onset
#     logging.info('epoching for {}'.format(dset))
#     logging.info('events for {0}: {1}'.format(dset,np.unique(dataEEG[dset]['events'][:,2], return_counts = True)))
#     np.save(savePath + dset + '_events', dataEEG[dset]['events'])
#     dataEEG[dset]['epochs'] = mne.Epochs(dataEEG[dset]['interpolated'], dataEEG[dset]['events'], event_id=100, tmin=-.250, tmax=2., baseline=(-.1,0), preload=True, reject=None, flat=None, reject_by_annotation=None, detrend=1)
#
#     # removing bad epochs
#     logging.info('removing bad epochs for {}'.format(dset))
#     badEpochs[dset] = eegfun.findBadEpochs(dataEEG[dset]['epochs']) # identifying bad epochs using FASTER
#
#     # re-sampling to speed-up computations
#     logging.info('re-sampling at 256 Hz for {}'.format(dset))
#     dataEEG[dset]['epochsSS'] = dataEEG[dset]['epochs'].copy().resample(256, npad='auto')
#     dataEEG[dset]['epochsSS'] = eegfun.dropBadEpochs(dataEEG[dset]['epochsSS'], badEpochs[dset])
#
#     # running ICA and excluding artifact components
#     logging.info('running ICA data for {}'.format(dset))
#     dataEEG[dset]['ica'] = eegfun.runICA(dataEEG[dset]['epochsSS'])
#     dataEEG[dset]['ica'].exclude = eegfun.findBadICA(dataEEG[dset]['epochsSS'], dataEEG[dset]['ica'])
#     badICA[dset] = dataEEG[dset]['ica'].exclude
#     dataEEG[dset]['epochs'] = eegfun.dropBadICA(dataEEG[dset]['epochs'], dataEEG[dset]['ica'])
#
#     # saving data
#     logging.info('saving data for {}'.format(dset))
#     dataEEG[dset]['ica'].save(savePath + dset + '_ica.fif.gz')
#     dataEEG[dset]['epochs'].save(savePath + dset + '_epochs.fif.gz')
#     pickle.dump({'chnnls':badChannels[dset],'epochs':badEpochs[dset],'icas':badICA[dset]}, open(savePath + dset + '_bads.pkl', 'wb'))
#===============================================================================
# Finding outlier participants
#===============================================================================
# %%
# dataEEG = OrderedDict()
# # dset = dsets[0]
# for dset in dsets:
#     dataEEG[dset] = {}
#     badEpochs  = pickle.load(open(savePath + dset + '_bads.pkl', 'rb'))['epochs']
#     dataEEG[dset]['badEpochs'] = badEpochs
#     dataEEG[dset]['erps'] = eegfun.dropBadEpochs(mne.epochs.read_epochs(savePath + dset + '_epochs.fif.gz'), badEpochs).average()
#
# picks = mne.pick_types(dataEEG[dset]['erps'].info, eeg=True, eog=True)
# erps = np.array([dataEEG[dset]['erps'].pick_types(eeg = True, eog = True).data for dset in dataEEG])
# idx_badSets = eegfun.findBadSets(erps)
# bad_sets = np.array(dsets)[idx_badSets]
# the first three are EEG outliers, the middle is accuracy outlier, the last RT
# bad_sets = ['s01','s10','s25'] + ['s22', 's30'] + ['s20']
bad_sets = ['s01','s10','s25']
good_sets = [dset for dset in dsets if dset not in bad_sets]
# good_sets = ['s35','s36'] # testing sets for qbi-uqdrange
#===============================================================================
# reading behavioural data
#===============================================================================
# %%
dataEEG = {}
for dset in good_sets:
    dataEEG[dset] = {}

    # loading behavioural data
    dataEEG[dset]['bhv'] = pd.read_csv(bhvPath + dset + '_bhv_data.txt', sep = '\t')
    idx_Epochs = np.where(dataEEG[dset]['bhv']['goodTrials'] == 0)[0]
    if dset == 's31':
        idx_Epochs = np.unique(np.concatenate([idx_Epochs, np.arange(192,256)]))
    dataEEG[dset]['bhv'].drop(idx_Epochs, inplace = True)
    dataEEG[dset]['bhv'].reset_index(inplace = True)

    # collating bad trials (EEG, dropped frames and missing responses)
    dataEEG[dset]['bads'] = pickle.load(open(savePath + dset + '_bads.pkl', 'rb'))
    # dataEEG[dset]['bads'] = pickle.load(open(savePath + dset + '_bads_2.pkl', 'rb')) # files with protocol 2
    dropdFrames = np.where(dataEEG[dset]['bhv']['dropdFrames'] != 0)[0]
    missedResp = np.where(dataEEG[dset]['bhv']['response'].isnull())[0]
    dataEEG[dset]['dropd'] = np.unique(np.hstack([dataEEG[dset]['bads']['epochs'],
                                                  dropdFrames,
                                                  missedResp]))

    dataEEG[dset]['bhv'].drop(dataEEG[dset]['dropd'], inplace = True)

    # encoding target motion direction in radians
    dataEEG[dset]['bhv']['tarDirRad'] = np.deg2rad(mm.wrap(dataEEG[dset]['bhv']['tarDir'], 180))
    # encoding experimental conditions
    dataEEG[dset]['bhv']['decSig'] = dataEEG[dset]['bhv']['diffOri'].abs()
    dataEEG[dset]['bhv']['decAlt'] = np.sign(dataEEG[dset]['bhv']['diffOri'])
    dataEEG[dset]['bhv']['rspAlt'] = dataEEG[dset]['bhv']['response'].astype('category').cat.codes

    dataEEG[dset]['idxSig'] = dataEEG[dset]['bhv']['decSig'].values == 15
    dataEEG[dset]['idxAlt'] = dataEEG[dset]['bhv']['decAlt'].values < 10
# #===============================================================================
# # %% loading EEG data
# #===============================================================================
# elocs = np.deg2rad(np.loadtxt(montagePath + 'biosemi64_QBI.txt',
#                               delimiter = '\t', skiprows = 1, usecols= [1,2])).transpose()
# for dset in good_sets:
#     # loading EEG data
#     dataEEG[dset]['epochs'] = mne.epochs.read_epochs(savePath + dset + '_epochs.fif.gz')
#     dataEEG[dset]['epochs'].resample(256)
#     # dropping bad trials
#     dataEEG[dset]['epochs'].drop(dataEEG[dset]['dropd'])

# picks = mne.pick_types(dataEEG[dset]['epochs'].info, eeg=True)
# chnames = np.array(dataEEG[dset]['epochs'].ch_names)[picks]
# times = dataEEG[dset]['epochs'].times
#===============================================================================
# %% Analysing ERP data
#===============================================================================
# loading erps per participant
for dset in good_sets:
    try:
        epochs = mne.epochs.read_epochs(savePath + dset + '_epochs_256Hz.fif.gz')
    except:
        epochs = mne.epochs.read_epochs(savePath + dset + '_epochs.fif.gz')
        epochs.resample(256)
    epochs = epochs.apply_baseline(baseline =(-.1, 0))
    # dropping bad trials
    epochs.drop(dataEEG[dset]['dropd'])
    # mne.preprocessing.compute_current_source_density(epochs, copy = False)
    epochs.pick(['eeg'])
    
    epochsStimLckd = epochs.get_data()
    ntrials, nelectrodes, nsamples = epochsStimLckd.shape 
    # response-locked epochs
    respOnsets = epochs.time_as_index(
        dataEEG[dset]['bhv']['RT']
    )
    epochsRespLckd = np.swapaxes(epochsStimLckd[
        np.arange(ntrials)[:, None], 
        :,
        # selecting period around response (-500 ms to 20 ms) 
        respOnsets[:, None] 
        + np.arange(-128, 5)[None]
    ], -1, 1)
    dataEEG[dset]['erpStimLckd'] = epochsStimLckd.mean(0)
    dataEEG[dset]['erpRespLckd'] = epochsRespLckd.mean(0)
    
    for idx_acc, acc in enumerate(['Cor', 'Err']):
        idx_trl = dataEEG[dset]['bhv']['error'] == idx_acc
        dataEEG[dset][f'erpStimLckd_{acc}'] = epochsStimLckd[idx_trl].mean(0)
        dataEEG[dset][f'erpRespLckd_{acc}'] = epochsRespLckd[idx_trl].mean(0)

    for idx_rt, rt in enumerate(['Fast', 'Mean', 'Slow']):
        if rt == 'Mean':
            continue
        else:
            idx_cor = dataEEG[dset]['bhv']['error'] == 0
            idx_trl = pd.qcut(
                dataEEG[dset]['bhv'].loc[idx_cor, 'RT'], 
                q = 3, 
                labels = False
            ).values == idx_rt
            dataEEG[dset][f'erpStimLckd_{rt}'] = epochsStimLckd[idx_cor][idx_trl].mean(0)
            dataEEG[dset][f'erpRespLckd_{rt}'] = epochsRespLckd[idx_cor][idx_trl].mean(0)

    for idx_diff, diff in enumerate([30, 15]):
        idx_cor = dataEEG[dset]['bhv']['error'] == 0
        idx_trl = dataEEG[dset]['bhv'].loc[idx_cor, 'decSig'] == diff
        diffLabel = 'Easy'
        if diff == 15:
            diffLabel = 'Hard'
        dataEEG[dset][f'erpStimLckd_{diffLabel}'] = epochsStimLckd[idx_cor][idx_trl].mean(0)
        dataEEG[dset][f'erpRespLckd_{diffLabel}'] = epochsRespLckd[idx_cor][idx_trl].mean(0)
# %%
montage = mne.channels.read_custom_montage(montagePath + 'biosemi64_QBI.txt')
chnls_roi = ['Pz', 'CP1', 'CPz', 'CP2', 'Cz']
idx_roi = [
    montage.ch_names.index(chnl)
    for chnl in chnls_roi
]

gavERP = dict([
    (
        f'gav_{lck}Lckd',
        np.array([
            dataEEG[dset][f'erp{lck}Lckd']
            for dset in good_sets
        ]).mean(0)
    )
    for lck in ['Stim', 'Resp']
])

# %% plot topomaps
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

figsize = (6.20, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)
# %%
times_StimLckd = np.linspace(-.25, 2, int(2.25 * 256))
times_RespLckd = np.linspace(-.5, .02, int(.52 * 256))
fig = plt.figure(figsize=figsize)
avEpochs = epochs.average()
avEpochs.data = gavERP['gav_StimLckd'].copy()
avEpochs.times = times_StimLckd
mask_roi = np.zeros([64, times_StimLckd.size])
mask_roi[idx_roi] = 1
mask_roi = mask_roi.astype('bool')
for idx_time, time in enumerate(np.linspace(0, 1, 5)):
    ax = fig.add_subplot(1, 9, idx_time + 1)
    avEpochs.plot_topomap(
        time, axes = ax,
        colorbar = False,
        vmin = -3, vmax = 3, 
        mask = mask_roi, 
        mask_params = dict(
            markersize = 6,
            marker = 'o',
            markerfacecolor = 'w',
            markeredgecolor = 'k'
        ),
        time_format = ''
        # time_format = "%0.2f s"
    )
avEpochs = epochs.average()
avEpochs.data = gavERP['gav_RespLckd'].copy()
avEpochs.times = times_RespLckd
for idx_time, time in enumerate(np.linspace(-.5, 0, 3)):
    ax = fig.add_subplot(1, 9, idx_time + 7)
    if idx_time + 1 == 2:
        ax.set_title('Resp-locked')
    avEpochs.plot_topomap(
        time, axes = ax,
        colorbar = False,
        vmin = -3, vmax = 3, 
        mask = mask_roi, 
        mask_params = dict(
            markersize = 6,
            marker = 'o',
            markerfacecolor = 'w',
            markeredgecolor = 'k'
        ),
        time_format = ''
        # time_format = "%0.2f s"
    )
fig.tight_layout()
# add titles
ax = fig.axes[2]
ax.text(
    .5, 1.5, 'Stimulus-locked',
    transform = ax.transAxes,
    ha = 'center', va = 'bottom',
    fontsize = MSIZE
)
ax = fig.axes[-2]
ax.text(
    .5, 1.5, 'Response-locked',
    transform = ax.transAxes,
    ha = 'center', va = 'bottom',
    fontsize = MSIZE
)
# add time annotations
for idx_time, time in enumerate(np.linspace(0, 1, 5)):
    ax = fig.axes[idx_time]
    ax.text(
        .5, -.25, ['0 s', '.25', '.50', '.75', '1'][idx_time],
        transform = ax.transAxes,
        ha = 'center', va = 'top',
        fontsize = SSIZE
    )
for idx_time, time in enumerate(np.linspace(-.5, 0, 3)):
    ax = fig.axes[idx_time + 5]
    ax.text(
        .5, -.25, ['-.5', '-.25', '0 s'][idx_time],
        transform = ax.transAxes,
        ha = 'center', va = 'top',
        fontsize = SSIZE
    )
# add colorbar
ax = fig.add_subplot(1,9,6)
im = fig.axes[0].get_children()[-2]
cbar = plt.colorbar(
    im, ax = ax,
    orientation = 'horizontal',
    fraction = .5, aspect = 5
)
ax.set_visible(False)
cbar.set_ticks([-3,3])
cbar.set_ticklabels(['-3', r'3 $\mu$V'])
# %%
gavROI = dict([
    (
        f'gav_{lck}Lckd',
        np.array([
            dataEEG[dset][f'erp{lck}Lckd'][idx_roi].mean(0)
            for dset in good_sets
        ]).mean(0)
    )
    for lck in ['Stim', 'Resp']
])
semROI = dict([
    (
        f'sem_{lck}Lckd',
        np.array([
            dataEEG[dset][f'erp{lck}Lckd'][idx_roi].mean(0)
            for dset in good_sets
        ]).std(0) / np.sqrt(len(good_sets))
    )
    for lck in ['Stim', 'Resp']
])

t_roi_slckd, p_roi_slckd = stats.ttest_1samp(
    np.array([
        dataEEG[dset]['erpStimLckd'][idx_roi].mean(0)
        for dset in good_sets
    ]),
    0, 0
)
padj_roi_slckd = multitest.fdrcorrection(p_roi_slckd)[1]

t_roi_rlckd, p_roi_rlckd = stats.ttest_1samp(
    np.array([
        dataEEG[dset]['erpRespLckd'][idx_roi].mean(0)
        for dset in good_sets
    ]),
    0, 0
)
padj_roi_rlckd = multitest.fdrcorrection(p_roi_rlckd)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray

figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

times_StimLckd = np.linspace(-.25, 2, int(2.25 * 256))
times_RespLckd = np.linspace(-.5, .02, int(.52 * 256))
fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
axStim.fill_between(
    times_StimLckd,
    gavROI['gav_StimLckd'] - semROI['sem_StimLckd'],
    gavROI['gav_StimLckd'] + semROI['sem_StimLckd'],
    color = 'darkred',
    alpha = .3
)
axStim.plot(
    times_StimLckd,
    gavROI['gav_StimLckd'],
    color = 'darkred'
)
axStim.set_xlim(-.02, 1.02)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)
axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_roi_slckd / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'ROI vs 0'
)

axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-2.25, 4.25)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
axResp.fill_between(
    times_RespLckd,
    gavROI['gav_RespLckd'] - semROI['sem_RespLckd'],
    gavROI['gav_RespLckd'] + semROI['sem_RespLckd'],
    color = 'darkred',
    alpha = .3
)
axResp.plot(
    times_RespLckd,
    gavROI['gav_RespLckd'],
    color = 'darkred'
)
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_roi_rlckd / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'ROI vs 0'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.8e-6, 2.4e-6)
# axStim.set_ylim(-.5e-6, 1.5e-6)
axStim.set_yticks([0, 1e-6])
axStim.set_yticklabels(['0', '1'])
axStim.set_ylabel('ROI ERP [$\mu$V]')
axStim.set_xticks([0, .25, .5, .75, 1])
axStim.set_xticklabels(['0 s', '.25', '.50', '.75', '1'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, 1e-6)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.02, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'gavROI.png', dpi = 600)
plt.close()
# %% ROI per response accuracy
gavROI_acc = dict([
    (
        f'{lck}Lckd',
        np.array([
            np.array([
                dataEEG[dset][f'erp{lck}Lckd_Cor'][idx_roi].mean(0),
                dataEEG[dset][f'erp{lck}Lckd_Err'][idx_roi].mean(0)
            ])
            for dset in good_sets
        ]).mean(0)
    )
    for lck in ['Stim', 'Resp']
])
semROI_acc = dict([
    (
        f'{lck}Lckd',
        np.sqrt(
            np.array([
                np.array([
                    dataEEG[dset][f'erp{lck}Lckd_Cor'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Err'][idx_roi].mean(0)
                ]) 
                # normalize score per individual intercept
                - np.array([
                    dataEEG[dset][f'erp{lck}Lckd_Cor'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Err'][idx_roi].mean(0)
                ]).mean(0)[None]
                for dset in good_sets
            # add overall mean
            ] + np.array([
                [
                    dataEEG[dset][f'erp{lck}Lckd_Cor'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Err'][idx_roi].mean(0)
                ] for dset in good_sets
            ]).mean(0)[:]).var(0, ddof = 1) * 2 / len(good_sets)
        ) 
    )
    for lck in ['Stim', 'Resp']
])
t_roi_slckd_acc, p_roi_slckd_acc = stats.ttest_rel(
    *np.stack([
        [
            dataEEG[dset]['erpStimLckd_Cor'][idx_roi].mean(0),
            dataEEG[dset]['erpStimLckd_Err'][idx_roi].mean(0)
        ]
        for dset in good_sets
    ], 1),
    0
)
padj_roi_slckd_acc = multitest.fdrcorrection(p_roi_slckd_acc)[1]

t_roi_rlckd_acc, p_roi_rlckd_acc = stats.ttest_rel(
    *np.stack([
        [
            dataEEG[dset]['erpRespLckd_Cor'][idx_roi].mean(0),
            dataEEG[dset]['erpRespLckd_Err'][idx_roi].mean(0)
        ]
        for dset in good_sets
    ], 1),
    0
)
padj_roi_rlckd_acc = multitest.fdrcorrection(p_roi_rlckd_acc)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray

figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

times_StimLckd = np.linspace(-.25, 2, int(2.25 * 256))
times_RespLckd = np.linspace(-.5, .02, int(.52 * 256))
fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)

for idx_acc, acc in enumerate(['Cor', 'Err']): 
    axStim.fill_between(
        times_StimLckd,
        gavROI_acc['StimLckd'][idx_acc] - semROI_acc['StimLckd'][idx_acc],
        gavROI_acc['StimLckd'][idx_acc] + semROI_acc['StimLckd'][idx_acc],
        color = 'darkred',
        alpha = .3
    )
    axStim.plot(
        times_StimLckd,
        gavROI_acc['StimLckd'][idx_acc],
        color = 'darkred',
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_acc],
        label = ['Correct', 'Error'][idx_acc]
    )
axStim.set_xlim(-.02, 1.02)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)
axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_roi_slckd_acc / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Cor vs Err'
)

axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-6.89, 13)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_acc, acc in enumerate(['Cor', 'Err']): 
    axResp.fill_between(
        times_RespLckd,
        gavROI_acc['RespLckd'][idx_acc] - semROI_acc['RespLckd'][idx_acc],
        gavROI_acc['RespLckd'][idx_acc] + semROI_acc['RespLckd'][idx_acc],
        color = 'darkred',
        alpha = .3
    )
    axResp.plot(
        times_RespLckd,
        gavROI_acc['RespLckd'][idx_acc],
        color = 'darkred',
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_acc],
        label = ['Correct', 'Error'][idx_acc]
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_roi_rlckd_acc / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Cor vs Err'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.8e-6, 2.4e-6)
axStim.set_yticks([0, 1e-6])
axStim.set_yticklabels(['0', '1'])
axStim.set_ylabel('ROI ERP [$\mu$V]')
axStim.set_xticks([0, .25, .5, .75, 1])
axStim.set_xticklabels(['0 s', '.25', '.50', '.75', '1'])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False, bbox_transform = axStim.transData,
    handlelength = 1, handletextpad = .5,
    loc = 'upper left', bbox_to_anchor = (0, 2.4e-6)
)

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, 1e-6)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.02, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'gavROI_acc.png', dpi = 600)
plt.close()
# %% ROI per response speed
gavROI_rt = dict([
    (
        f'{lck}Lckd',
        np.array([
            np.array([
                dataEEG[dset][f'erp{lck}Lckd_Fast'][idx_roi].mean(0),
                dataEEG[dset][f'erp{lck}Lckd_Slow'][idx_roi].mean(0)
            ])
            for dset in good_sets
        ]).mean(0)
    )
    for lck in ['Stim', 'Resp']
])
semROI_rt = dict([
    (
        f'{lck}Lckd',
        np.sqrt(
            np.array([
                np.array([
                    dataEEG[dset][f'erp{lck}Lckd_Fast'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Slow'][idx_roi].mean(0)
                ]) 
                # normalize sFaste per individual intercept
                - np.array([
                    dataEEG[dset][f'erp{lck}Lckd_Fast'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Slow'][idx_roi].mean(0)
                ]).mean(0)[None]
                for dset in good_sets
            # add overall mean
            ] + np.array([
                [
                    dataEEG[dset][f'erp{lck}Lckd_Fast'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Slow'][idx_roi].mean(0)
                ] for dset in good_sets
            ]).mean(0)[:]).var(0, ddof = 1) * 2 / len(good_sets)
        ) 
    )
    for lck in ['Stim', 'Resp']
])
t_roi_slckd_rt, p_roi_slckd_rt = stats.ttest_rel(
    *np.stack([
        [
            dataEEG[dset]['erpStimLckd_Fast'][idx_roi].mean(0),
            dataEEG[dset]['erpStimLckd_Slow'][idx_roi].mean(0)
        ]
        for dset in good_sets
    ], 1),
    0
)
padj_roi_slckd_rt = multitest.fdrcorrection(p_roi_slckd_rt)[1]

t_roi_rlckd_rt, p_roi_rlckd_rt = stats.ttest_rel(
    *np.stack([
        [
            dataEEG[dset]['erpRespLckd_Fast'][idx_roi].mean(0),
            dataEEG[dset]['erpRespLckd_Slow'][idx_roi].mean(0)
        ]
        for dset in good_sets
    ], 1),
    0
)
padj_roi_rlckd_rt = multitest.fdrcorrection(p_roi_rlckd_rt)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray

figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

times_StimLckd = np.linspace(-.25, 2, int(2.25 * 256))
times_RespLckd = np.linspace(-.5, .02, int(.52 * 256))
fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)

for idx_rt, rt in enumerate(['Fast', 'Slow']): 
    axStim.fill_between(
        times_StimLckd,
        gavROI_rt['StimLckd'][idx_rt] - semROI_rt['StimLckd'][idx_rt],
        gavROI_rt['StimLckd'][idx_rt] + semROI_rt['StimLckd'][idx_rt],
        color = 'darkred',
        alpha = .3
    )
    axStim.plot(
        times_StimLckd,
        gavROI_rt['StimLckd'][idx_rt],
        color = 'darkred',
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_rt],
        label = ['Fast', 'Slow'][idx_rt]
    )
axStim.set_xlim(-.02, 1.02)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)
axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_roi_slckd_rt / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)

axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-3.71, 7)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_rt, rt in enumerate(['Fast', 'Slow']): 
    axResp.fill_between(
        times_RespLckd,
        gavROI_rt['RespLckd'][idx_rt] - semROI_rt['RespLckd'][idx_rt],
        gavROI_rt['RespLckd'][idx_rt] + semROI_rt['RespLckd'][idx_rt],
        color = 'darkred',
        alpha = .3
    )
    axResp.plot(
        times_RespLckd,
        gavROI_rt['RespLckd'][idx_rt],
        color = 'darkred',
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_rt],
        label = ['Fast', 'Slow'][idx_rt]
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_roi_rlckd_rt / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.8e-6, 2.4e-6)
axStim.set_yticks([0, 1e-6])
axStim.set_yticklabels(['0', '1'])
axStim.set_ylabel('ROI ERP [$\mu$V]')
axStim.set_xticks([0, .25, .5, .75, 1])
axStim.set_xticklabels(['0 s', '.25', '.50', '.75', '1'])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False, bbox_transform = axStim.transData,
    handlelength = 1, handletextpad = .5,
    loc = 'upper left', bbox_to_anchor = (0, 2.4e-6)
)

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, 1e-6)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.02, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'gavROI_rt.png', dpi = 600)
plt.close()
# %% ROI per difficulty
gavROI_diff = dict([
    (
        f'{lck}Lckd',
        np.array([
            np.array([
                dataEEG[dset][f'erp{lck}Lckd_Easy'][idx_roi].mean(0),
                dataEEG[dset][f'erp{lck}Lckd_Hard'][idx_roi].mean(0)
            ])
            for dset in good_sets
        ]).mean(0)
    )
    for lck in ['Stim', 'Resp']
])
semROI_diff = dict([
    (
        f'{lck}Lckd',
        np.sqrt(
            np.array([
                np.array([
                    dataEEG[dset][f'erp{lck}Lckd_Easy'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Hard'][idx_roi].mean(0)
                ]) 
                # normalize sEasye per individual intercept
                - np.array([
                    dataEEG[dset][f'erp{lck}Lckd_Easy'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Hard'][idx_roi].mean(0)
                ]).mean(0)[None]
                for dset in good_sets
            # add overall mean
            ] + np.array([
                [
                    dataEEG[dset][f'erp{lck}Lckd_Easy'][idx_roi].mean(0),
                    dataEEG[dset][f'erp{lck}Lckd_Hard'][idx_roi].mean(0)
                ] for dset in good_sets
            ]).mean(0)[:]).var(0, ddof = 1) * 2 / len(good_sets)
        ) 
    )
    for lck in ['Stim', 'Resp']
])
t_roi_slckd_diff, p_roi_slckd_diff = stats.ttest_rel(
    *np.stack([
        [
            dataEEG[dset]['erpStimLckd_Easy'][idx_roi].mean(0),
            dataEEG[dset]['erpStimLckd_Hard'][idx_roi].mean(0)
        ]
        for dset in good_sets
    ], 1),
    0
)
padj_roi_slckd_diff = multitest.fdrcorrection(p_roi_slckd_diff)[1]

t_roi_rlckd_diff, p_roi_rlckd_diff = stats.ttest_rel(
    *np.stack([
        [
            dataEEG[dset]['erpRespLckd_Easy'][idx_roi].mean(0),
            dataEEG[dset]['erpRespLckd_Hard'][idx_roi].mean(0)
        ]
        for dset in good_sets
    ], 1),
    0
)
padj_roi_rlckd_diff = multitest.fdrcorrection(p_roi_rlckd_diff)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray

figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

times_StimLckd = np.linspace(-.25, 2, int(2.25 * 256))
times_RespLckd = np.linspace(-.5, .02, int(.52 * 256))
fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)

for idx_diff, diff in enumerate(['Easy', 'Hard']): 
    axStim.fill_between(
        times_StimLckd,
        gavROI_diff['StimLckd'][idx_diff] - semROI_diff['StimLckd'][idx_diff],
        gavROI_diff['StimLckd'][idx_diff] + semROI_diff['StimLckd'][idx_diff],
        color = 'darkred',
        alpha = .3
    )
    axStim.plot(
        times_StimLckd,
        gavROI_diff['StimLckd'][idx_diff],
        color = 'darkred',
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_diff],
        label = ['Easy', 'Hard'][idx_diff]
    )
axStim.set_xlim(-.02, 1.02)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)
axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_roi_slckd_diff / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Easy vs Hard'
)

axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-3.71, 7)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_diff, diff in enumerate(['Easy', 'Hard']): 
    axResp.fill_between(
        times_RespLckd,
        gavROI_diff['RespLckd'][idx_diff] - semROI_diff['RespLckd'][idx_diff],
        gavROI_diff['RespLckd'][idx_diff] + semROI_diff['RespLckd'][idx_diff],
        color = 'darkred',
        alpha = .3
    )
    axResp.plot(
        times_RespLckd,
        gavROI_diff['RespLckd'][idx_diff],
        color = 'darkred',
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_diff],
        label = ['Easy', 'Hard'][idx_diff]
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_roi_rlckd_diff / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Easy vs Hard'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.8e-6, 2.4e-6)
axStim.set_yticks([0, 1e-6])
axStim.set_yticklabels(['0', '1'])
axStim.set_ylabel('ROI ERP [$\mu$V]')
axStim.set_xticks([0, .25, .5, .75, 1])
axStim.set_xticklabels(['0 s', '.25', '.50', '.75', '1'])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False, bbox_transform = axStim.transData,
    handlelength = 1, handletextpad = .5,
    loc = 'upper left', bbox_to_anchor = (0, 2.4e-6)
)

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, 1e-6)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.02, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'gavROI_diff.png', dpi = 600)
plt.close()
#===============================================================================
# %% Plotting ERP data
#===============================================================================
# # computing erps per participant
# for dset in good_sets:
#     try:
#         epochs = mne.epochs.read_epochs(savePath + dset + '_epochs_256Hz.fif.gz')
#     except:
#         epochs = mne.epochs.read_epochs(savePath + dset + '_epochs.fif.gz')
#         epochs.resample(256)
#     epochs = epochs.apply_baseline(baseline =(-.1, 0))
#     # dropping bad trials
#     epochs.drop(dataEEG[dset]['dropd'])
#     idx_chnls = mne.pick_types(
#         epochs.info,
#         eeg = True,
#         eog = True
#     )
#     epochs_data = epochs.get_data()
#     ntrials, nelectrodes, nsamples = epochs_data.shape 
#     dataEEG[dset]['erpStimLckd'] = epochs_data.mean(0)[idx_chnls]
    
#     respOnsets = epochs.time_as_index(
#             dataEEG[dset]['bhv']['RT']
#         )
#     dataEEG[dset]['erpRespLckd'] = epochs_data[
#         np.arange(ntrials)[:, None], 
#         :,
#         # selecting period around response (-500 ms to 20 ms) 
#         respOnsets[:, None] 
#         + np.arange(-128, 5)[None]
#     ].mean(0).T[idx_chnls]
# # saving the erps
# with h5py.File(exportPath + 'aggERP.hdf', 'w') as f:
#     f.create_dataset(
#         name = 'erpStimLckd',
#         data = np.array([
#             dataEEG[dset]['erpStimLckd'] 
#             for dset in good_sets
#         ]),
#         compression = 9
#     )
#     f['erpStimLckd'].attrs['chnames'] = list(np.array(epochs.ch_names)[idx_chnls])
#     f['erpStimLckd'].attrs['dsets'] = list(dataEEG.keys())
#     f.create_dataset(
#         name = 'erpRespLckd',
#         data = np.array([
#             dataEEG[dset]['erpRespLckd'] 
#             for dset in good_sets
#         ]),
#         compression = 9
#     )
#     f['erpRespLckd'].attrs['chnames'] = list(np.array(epochs.ch_names)[idx_chnls])
#     f['erpRespLckd'].attrs['dsets'] = list(dataEEG.keys())

with h5py.File(exportPath + 'aggERP.hdf', 'r') as f:
    aggERP_stimLckd = f['erpStimLckd'][:]
    chnames = f['erpStimLckd'].attrs['chnames']
    aggERP_respLckd = f['erpRespLckd'][:]

tERP_stimLckd, pERP_stimLckd = stats.ttest_1samp(aggERP_stimLckd, 0)
padjERP_stimLckd = multitest.fdrcorrection(pERP_stimLckd.flatten())[-1].reshape(
    *pERP_stimLckd.shape
)

gavERP_stimLckd = aggERP_stimLckd.mean(0)
gavERP_respLckd = aggERP_respLckd.mean(0)

times_StimLckd = np.linspace(-.25, 2, int(2.25 * 256))
times_RespLckd = np.linspace(-.5, .02, int(.52 * 256))
# %% plotting ERPs
# montage = mne.channels.read_montage(kind = 'biosemi64_QBI',
#                                     path = montagePath)
montage = mne.channels.read_custom_montage(montagePath + 'biosemi64_QBI.txt')

# pos = montage.get_pos2d()
pos = np.array([v for v in montage._get_ch_pos().values()])
idx_pos = pos[:, 1].argsort()

sbn.set_style('ticks')
sbn.set_palette('husl', 75, desat = .8)
ln_col = '#3F3F3F' # dark gray

figsize = (2.7, 1.35)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (1, 4), (0, 0), colspan = 3
)
axStim.plot(
    times_StimLckd,
    mne.filter.filter_data(
        gavERP_stimLckd[:-2],
        256, None, 10
    )[idx_pos].T,
    lw = .75
)
# for idx_eog, eog in enumerate(['hEOG', 'vEOG']):
#     axStim.plot(
#         times_StimLckd,
#         mne.filter.filter_data(
#             gavERP_stimLckd[-2:],
#             256, None, 10
#         )[idx_eog].T,
#         dashes = [
#             (1, 0),
#             (1, 1)
#         ][idx_eog],
#         color = ln_col,
#         label = eog,
#     )
axStim.set_xlim(-.02, 1)
axStim.set_xticks(np.linspace(0, 1, 5))
axStim.set_xticklabels(['0 s', '.25', '.50', '.75', '1'])
axStim.spines['bottom'].set_bounds(0, 1)
axStim.set_title('Stimulus-locked', fontsize = SSIZE)

axResp = plt.subplot2grid(
    (1, 4), (0, 3), colspan = 1,
    sharey = axStim
)
axResp.plot(
    times_RespLckd,
    mne.filter.filter_data(
        gavERP_respLckd[:-2],
        256, None, 10
    )[idx_pos].T,
    lw = .75
)
# for idx_eog, eog in enumerate(['hEOG', 'vEOG']):
#     axResp.plot(
#         times_RespLckd,
#         mne.filter.filter_data(
#             gavERP_respLckd[-2:],
#             256, None, 10
#         )[idx_eog].T,
#         dashes = [
#             (1, 0),
#             (1, 1)
#         ][idx_eog],
#         color = ln_col,
#         label = eog,
#     )
# axResp.axhline(-np.log10(.05), color = 'gray')
axResp.set_xlim(-.5, .02)
axResp.set_xticks(np.linspace(-.5, 0, 3))
axResp.set_xticklabels(['-.50', '-.25', '0 s'])
axResp.spines['bottom'].set_bounds(-.5, 0)
axResp.set_title('Response-locked', fontsize = SSIZE)

for ax in fig.axes:
    ax.yaxis.set_visible(False)
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
fig.tight_layout(rect = (0, 0, 1, 1), pad = .15)
plt.subplots_adjust(wspace = .4)

axTopo = fig.add_axes([.6, .6, .25, .25])
axTopo.set_aspect('equal')
pos, outlines = mne.viz.topomap._check_outlines(
    pos[idx_pos],
    np.array([0.7, 0.7]),
    {
        'center': (0, 0),
        'scale': (.4, .4)
    }
)
mne.viz.topomap._draw_outlines(axTopo, outlines)
axTopo.scatter(
    pos[:, 0], pos[:, 1], 
    picker=True, s=2,
    color = sbn.husl_palette(75, s = .8)[:64],
    edgecolor='None', linewidth=1, clip_on=False
)
axTopo.scatter(
    pos[idx_pos, 0], pos[idx_pos, 1], 
    picker=True, s=2,
    color = sbn.husl_palette(75, s = .8)[:64],
    edgecolor='None', linewidth=1, clip_on=False
)
for spine in ['left', 'top', 'right', 'bottom']:
    axTopo.spines[spine].set_visible(False)
axTopo.xaxis.set_visible(False)
axTopo.yaxis.set_visible(False)

hdls, lbls = axStim.get_legend_handles_labels()
# axStim.legend(
#     hdls, lbls,
#     loc = 4, bbox_to_anchor = (.675, .55), bbox_transform = fig.transFigure,
#     frameon = False,
#     handlelength = .9, handletextpad = .5
# )
# fig.savefig(exportPath + 'gavERP_allElectrodes.png', dpi = 600)
# plt.close()
#===============================================================================
# %% Analysing electrode sensitivity to motion direction and response alternative
#===============================================================================
elocs = np.deg2rad(np.loadtxt(
    montagePath + 'biosemi64_QBI.txt',
    delimiter = '\t', skiprows = 1, usecols= [1,2]
)).transpose()

def time_as_index(alltimes, tartimes):
    return np.argmin(np.abs(np.array(
        alltimes)[None] - np.array(tartimes)[:,None]
    ),1)
alltimes = np.linspace(-.25, 2, int(2.25 * 256))

aggStimLckd_Dir = []
aggRespLckd_Dir = []
aggStimLckd_Alt = []
aggRespLckd_Alt = []
for dset in dataEEG:
    with h5py.File(savePath + '{}_electrodeSensitivity.hdf'.format(dset), 'r') as f:
        aggStimLckd_Dir += [f['zTarDir_Stim'][:]]
        aggRespLckd_Dir += [f['zTarDir_Resp'][:]]
        aggStimLckd_Alt += [f['zDecAlt_Stim'][:]]
        aggRespLckd_Alt += [f['zDecAlt_Resp'][:]]
        
        chnames = f['zTarDir_Stim'].attrs['chnames']
        times_StimLckd = f['zTarDir_Stim'].attrs['times']
        times_RespLckd = f['zTarDir_Resp'].attrs['times']

SD = int(.016 * 256)
gwin_Stim = scipy.signal.gaussian(int(2.250 * 256), SD)
gwin_Stim /= gwin_Stim.sum()
gwin_Resp = scipy.signal.gaussian(int(.52 * 256), SD)
gwin_Resp /= gwin_Resp.sum()

for dset in dataEEG:
    dataEEG[dset]['chrspStm'] = scipy.signal.convolve(dataEEG[dset]['chrspStm'],
                                                   gwin[None, None],
                                                   mode = 'same')

aggStimLckd_Dir = scipy.signal.convolve(
    aggStimLckd_Dir,
    gwin_Stim[None, None],
    mode = 'same'
)
aggRespLckd_Dir = np.array(aggRespLckd_Dir)
aggStimLckd_Alt = np.array(aggStimLckd_Alt)
aggRespLckd_Alt = np.array(aggRespLckd_Alt)
# replacing infite z values
z_aggStimLckd_Dir = stats.norm.ppf(aggStimLckd_Dir)
z_aggStimLckd_Dir[np.isneginf(
    z_aggStimLckd_Dir
)] = -8.21
z_aggStimLckd_Dir[np.isinf(
    z_aggStimLckd_Dir
)] = 8.21

# motion direction
# stimLckd_Dir_tval, stimLckd_Dir_pval = stats.ttest_1samp(
#     aggStimLckd_Dir - aggStimLckd_Dir[..., :64].mean(-1)[..., None],
#     0
# )
stimLckd_Dir_tval, stimLckd_Dir_pval = stats.ttest_1samp(
    aggStimLckd_Dir - .5, 0
)
stimLckd_Dir_tval, stimLckd_Dir_pval = stats.ttest_1samp(
    z_aggStimLckd_Dir, 0
)
stimLckd_Dir_padj = multitest.fdrcorrection(
    stimLckd_Dir_pval.flatten()
)[1].reshape(*stimLckd_Dir_pval.shape)

# respLckd_Dir_tval, respLckd_Dir_pval = stats.ttest_1samp(
#     aggRespLckd_Dir - aggStimLckd_Dir[..., :64].mean(-1)[..., None],
#     0
# )
respLckd_Dir_tval, respLckd_Dir_pval = stats.ttest_1samp(
    aggRespLckd_Dir, .5
)
respLckd_Dir_padj = multitest.fdrcorrection(
    respLckd_Dir_pval.flatten()
)[1].reshape(*respLckd_Dir_pval.shape)

# decision alternative
# stimLckd_Alt_tval, stimLckd_Alt_pval = stats.ttest_1samp(
#     aggStimLckd_Alt - aggStimLckd_Alt[..., :64].mean(-1)[..., None],
#     0
# )
stimLckd_Alt_tval, stimLckd_Alt_pval = stats.ttest_1samp(
    aggStimLckd_Alt, .5
)
stimLckd_Alt_padj = multitest.fdrcorrection(
    stimLckd_Alt_pval.flatten()
)[1].reshape(*stimLckd_Alt_pval.shape)

# respLckd_Alt_tval, respLckd_Alt_pval = stats.ttest_1samp(
#     aggRespLckd_Alt - aggStimLckd_Alt[..., :64].mean(-1)[..., None],
#     0
# )
respLckd_Alt_tval, respLckd_Alt_pval = stats.ttest_1samp(
    aggRespLckd_Alt, .5
)
respLckd_Alt_padj = multitest.fdrcorrection(
    respLckd_Alt_pval.flatten()
)[1].reshape(*respLckd_Alt_pval.shape)

# %% plotting electrode sensitivity
montage = mne.channels.read_montage(kind = 'biosemi64_QBI',
                                    path = montagePath)
pos = montage.get_pos2d()
idx_pos = pos[:, 1].argsort()

sbn.set_style('ticks')
sbn.set_palette('husl', 75, desat = .8)
ln_col = '#3F3F3F' # dark gray

figsize = (2.7, 1.35)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (1, 4), (0, 0), colspan = 3
)
axStim.plot(
    times_StimLckd,
    mne.filter.filter_data(
        -np.log10(stimLckd_Dir_padj[:-2]),
        256, None, 10
    )[idx_pos].T,
    lw = .75
)
for idx_eog, eog in enumerate(['hEOG', 'vEOG']):
    axStim.plot(
        times_StimLckd,
        mne.filter.filter_data(
            -np.log10(stimLckd_Dir_padj[-2:]),
            256, None, 10
        )[idx_eog].T,
        dashes = [
            (1, 0),
            (1, 1)
        ][idx_eog],
        color = ln_col,
        label = eog,
    )
axStim.axhline(-np.log10(.05), color = 'gray')
axStim.set_xlim(-.02, 1.5)
axStim.set_xticks(np.linspace(0, 1.5, 4))
axStim.set_xticklabels(['0 s', '.5', '1.0', '1.5'])
axStim.spines['bottom'].set_bounds(0, 1.5)
axStim.set_title('Stimulus-locked', fontsize = SSIZE)
axStim.text(
    1.5, -np.log10(.05),
    r'$-$log$_{10}$(p=.05)',
    ha = 'right', va = 'bottom',
    fontsize = SSIZE
)

axResp = plt.subplot2grid(
    (1, 4), (0, 3), colspan = 1,
    sharey = axStim
)
axResp.plot(
    times_RespLckd,
    mne.filter.filter_data(
        -np.log10(respLckd_Dir_padj[:-2]),
        256, None, 10
    )[idx_pos].T,
    lw = .75
)
for idx_eog, eog in enumerate(['hEOG', 'vEOG']):
    axResp.plot(
        times_RespLckd,
        mne.filter.filter_data(
            -np.log10(respLckd_Dir_padj[-2:]),
            256, None, 10
        )[idx_eog].T,
        dashes = [
            (1, 0),
            (1, 1)
        ][idx_eog],
        color = ln_col,
        label = eog,
    )
axResp.axhline(-np.log10(.05), color = 'gray')
axResp.set_xlim(-.5, .02)
axResp.set_xticks(np.linspace(-.5, 0, 2))
axResp.set_xticklabels(['-.5', '0 s'])
axResp.spines['bottom'].set_bounds(-.5, 0)
axResp.set_title('Response-locked', fontsize = SSIZE)

for ax in fig.axes:
    ax.yaxis.set_visible(False)
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
fig.tight_layout(rect = (0, 0, 1, 1), pad = .15)
plt.subplots_adjust(wspace = .4)

axTopo = fig.add_axes([.6, .6, .25, .25])
axTopo.set_aspect('equal')
pos, outlines = mne.viz.topomap._check_outlines(
    pos[idx_pos],
    np.array([0.7, 0.7]),
    {
        'center': (0, 0),
        'scale': (.4, .4)
    }
)
mne.viz.topomap._draw_outlines(axTopo, outlines)
axTopo.scatter(
    pos[:, 0], pos[:, 1], 
    picker=True, s=2,
    color = sbn.husl_palette(75, s = .8)[:64],
    edgecolor='None', linewidth=1, clip_on=False
)
for spine in ['left', 'top', 'right', 'bottom']:
    axTopo.spines[spine].set_visible(False)
axTopo.xaxis.set_visible(False)
axTopo.yaxis.set_visible(False)

hdls, lbls = axStim.get_legend_handles_labels()
axStim.legend(
    hdls, lbls,
    loc = 4, bbox_to_anchor = (.675, .55), bbox_transform = fig.transFigure,
    frameon = False,
    handlelength = .9, handletextpad = .5
)
fig.savefig(exportPath + 'electrodeSensitivity_motDir.png', dpi = 600)
plt.close()
# %% plotting electrode sensitivity
montage = mne.channels.read_montage(kind = 'biosemi64_QBI',
                                    path = montagePath)
pos = montage.get_pos2d()
idx_pos = pos[:, 1].argsort()

sbn.set_style('ticks')
sbn.set_palette('husl', 75, desat = .8)
ln_col = '#3F3F3F' # dark gray

figsize = (2.7, 1.35)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (1, 4), (0, 0), colspan = 3
)
axStim.plot(
    times_StimLckd,
    mne.filter.filter_data(
        -np.log10(stimLckd_Alt_padj[:-2]),
        256, None, 10
    )[idx_pos].T,
    lw = .75
)
for idx_eog, eog in enumerate(['hEOG', 'vEOG']):
    axStim.plot(
        times_StimLckd,
        mne.filter.filter_data(
            -np.log10(stimLckd_Alt_padj[-2:]),
            256, None, 10
        )[idx_eog].T,
        dashes = [
            (1, 0),
            (1, 1)
        ][idx_eog],
        color = ln_col,
        label = eog,
    )
axStim.set_ylim(-0.8253493074036371, 12.705938165993283)
axStim.axhline(-np.log10(.05), color = 'gray')
axStim.set_xlim(-.02, 1.5)
axStim.set_xticks(np.linspace(0, 1.5, 4))
axStim.set_xticklabels(['0 s', '.5', '1.0', '1.5'])
axStim.spines['bottom'].set_bounds(0, 1.5)
axStim.set_title('Stimulus-locked', fontsize = SSIZE)
axStim.text(
    1.5, -np.log10(.05),
    r'$-$log$_{10}$(p=.05)',
    ha = 'right', va = 'bottom',
    fontsize = SSIZE
)

axResp = plt.subplot2grid(
    (1, 4), (0, 3), colspan = 1,
    sharey = axStim
)
axResp.plot(
    times_RespLckd,
    mne.filter.filter_data(
        -np.log10(respLckd_Alt_padj[:-2]),
        256, None, 10
    )[idx_pos].T,
    lw = .75
)
for idx_eog, eog in enumerate(['hEOG', 'vEOG']):
    axResp.plot(
        times_RespLckd,
        mne.filter.filter_data(
            -np.log10(respLckd_Alt_padj[-2:]),
            256, None, 10
        )[idx_eog].T,
        dashes = [
            (1, 0),
            (1, 1)
        ][idx_eog],
        color = ln_col,
        label = eog,
    )
axResp.axhline(-np.log10(.05), color = 'gray')
axResp.set_xlim(-.5, .02)
axResp.set_xticks(np.linspace(-.5, 0, 2))
axResp.set_xticklabels(['-.5', '0 s'])
axResp.spines['bottom'].set_bounds(-.5, 0)
axResp.set_title('Response-locked', fontsize = SSIZE)

for ax in fig.axes:
    ax.yaxis.set_visible(False)
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
fig.tight_layout(rect = (0, 0, 1, 1), pad = .15)
plt.subplots_adjust(wspace = .4)

axTopo = fig.add_axes([.6, .6, .25, .25])
axTopo.set_aspect('equal')
pos, outlines = mne.viz.topomap._check_outlines(
    pos[idx_pos],
    np.array([0.7, 0.7]),
    {
        'center': (0, 0),
        'scale': (.4, .4)
    }
)
mne.viz.topomap._draw_outlines(axTopo, outlines)
axTopo.scatter(
    pos[:, 0], pos[:, 1], 
    picker=True, s=2,
    color = sbn.husl_palette(75, s = .8)[:64],
    edgecolor='None', linewidth=1, clip_on=False
)
for spine in ['left', 'top', 'right', 'bottom']:
    axTopo.spines[spine].set_visible(False)
axTopo.xaxis.set_visible(False)
axTopo.yaxis.set_visible(False)

hdls, lbls = axStim.get_legend_handles_labels()
axStim.legend(
    hdls, lbls,
    loc = 4, bbox_to_anchor = (.675, .55), bbox_transform = fig.transFigure,
    frameon = False,
    handlelength = .9, handletextpad = .5
)
fig.savefig(exportPath + 'electrodeSensitivity_rspAlt.png', dpi = 600)
plt.close()
#===============================================================================
# %% FORWARD ENCODING EEG
#===============================================================================
# DONE: save the cross-validation folds for future reference
# NOTE: using different python versions results in different shuffled sequences
# even when the same seed is used
# for dset in good_sets:
#     with h5py.File(
#         '{}{}_frw_eeg_All_empChnls.h5'.format(savePath, dset),
#         'r'
#     ) as f:
#         allTST = dict([
#             (fold_key, fold_val['TST'][:]) 
#             for fold_key, fold_val in f['root'].items()
#         ])
#     allTRLS = set(np.sort(np.concatenate([TST for TST in allTST.values()])))
#     CV = [
#         [np.array(list(allTRLS - set(TST))), TST]
#         for TST in allTST.values()
#     ]
#     with h5py.File('{}frwCV.hdf'.format(savePath), 'a') as f:
#         dset_data = f.create_group(dset)
#         for idx_fold, fold in enumerate(CV):
#             TRN, TST = fold
#             dset_fold = dset_data.create_group('fold_{}'.format(idx_fold))
#             dset_fold.create_dataset(
#                 name = 'TRN',
#                 data = TRN,
#                 compression = 9
#             )
#             dset_fold.create_dataset(
#                 name = 'TST',
#                 data = TST,
#                 compression = 9
#             )
# Computing distributions of errors per fold
cv_data = h5py.File('{}frwCV.hdf'.format(savePath), 'r')
# %%
[
    [(
        'fold_ALL', 
        [
            dataEEG[sub_key]['bhv']['error'].mean().round(2), 
            dataEEG[sub_key]['bhv']['error'].size
        ]
    )] 
    + [(
        fold_key, 
        [
            dataEEG[sub_key]['bhv'].iloc[fold_val['TRN']]['error'].mean().round(2),
            dataEEG[sub_key]['bhv'].iloc[fold_val['TRN']]['error'].size,
            dataEEG[sub_key]['bhv'].iloc[fold_val['TST']]['error'].size
        ]
    ) for fold_key, fold_val in sub_val.items()] 
    for sub_key, sub_val in cv_data.items()
]
#===============================================================================
# %% Analyzing channel responses
#===============================================================================
times_StimLckd= np.linspace(-.250, 2, int(2.250*256))
times_RespLckd = np.linspace(-.500, .020, int(.520*256))
# temporal smoothing parameters
SD = int(.016 * 256)
gwins = {}
for idx_epochs, duration_epochs in enumerate([2.250, .520]):
    gwin = scipy.signal.gaussian(int(duration_epochs * 256), SD)
    gwin /= gwin.sum()
    gwins[
        ['stimulus-locked', 'response-locked'][idx_epochs]
    ] = gwin

# loading channel responses
for dset in good_sets:
    logging.warning('Loading channel responses for {}'.format(dset))
    fname = savePath + dset + '_channelResponses.hdf'
    dataEEG[dset]['chRspSrtdSmth'] = {}
    with h5py.File(fname, 'r') as f:
        for key_name, key_value in f.items():
            channels = key_value['channelResponses'].attrs['channels'][:]
            chRsp = key_value['channelResponses'][:]
            srtIdx = key_value['channelSortIndex'][:]
            chRspSrtd = chRsp[
                np.arange(len(chRsp))[:, None], 
                srtIdx
            ]
            # smoothing channel responses
            chRspSrtdSmth = scipy.signal.convolve(
                chRspSrtd,
                gwins[key_name][None, None],
                mode = 'same'
            )
            dataEEG[dset]['chRspSrtdSmth'][key_name] = chRspSrtdSmth
channels = channels[0]
# %% aggregating responses
aggChrsp_stimLckd, aggChrsp_respLckd = [
    np.array([
        dataEEG[dset]['chRspSrtdSmth'][epochs].mean(0)
        for dset in dataEEG]) 
    for epochs in ['stimulus-locked', 'response-locked']
]
NBINS = 5
bins = pd.cut(
    pd.Series(channels), 
    NBINS, labels=np.arange(NBINS),
    retbins=False
).values
binCentres =np.rad2deg([
    channels[bins == bin].mean() 
    for bin in bins.unique()
]).round(1)

aggChrsp_stimLckd_binned = np.stack([
    aggChrsp_stimLckd[:, bins == bin].mean(1)
    for bin in bins.unique()
], 1)
aggChrsp_respLckd_binned = np.stack([
    aggChrsp_respLckd[:, bins == bin].mean(1)
    for bin in np.arange(5)
], 1)

gavChrsp_stimLckd_binned = aggChrsp_stimLckd_binned.mean(0)
gavChrsp_respLckd_binned = aggChrsp_respLckd_binned.mean(0)

semChresp_stimLckd_binned = np.sqrt((
    aggChrsp_stimLckd_binned
    - aggChrsp_stimLckd_binned.mean(1)[:, None]
    + gavChrsp_stimLckd_binned[None]
).var(0, ddof =1) * (4 / 3) / (aggChrsp_stimLckd_binned.shape[0] - 1))
semChresp_respLckd_binned = np.sqrt((
    aggChrsp_respLckd_binned
    - aggChrsp_respLckd_binned.mean(1)[:, None]
    + gavChrsp_respLckd_binned[None]
).var(0, ddof =1) * (4 / 3) / (aggChrsp_respLckd_binned.shape[0] - 1))

# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
# cols = np.array(sbn.husl_palette(NBINS, h = .3, s = .75))
cols = np.array(
    sbn.husl_palette(12, h = .7, s = .75)[1:3][::-1]
    + sbn.husl_palette(12, h = .7, s = .75)[:3]
)

figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 4
)
for idx_bin, bin in enumerate(binCentres):
    lbl = f'$\pm${np.abs(bin)}$\degree$'
    if bin == 0:
        lbl = f'{np.abs(bin)}$\degree$'
    axStim.fill_between(
        times_StimLckd, 
        gavChrsp_stimLckd_binned[idx_bin] - semChresp_stimLckd_binned[idx_bin], 
        gavChrsp_stimLckd_binned[idx_bin] + semChresp_stimLckd_binned[idx_bin],
        alpha = .3, color = cols[idx_bin]
    )
    axStim.plot(
        times_StimLckd,
        gavChrsp_stimLckd_binned[idx_bin],
        color = cols[idx_bin], label = lbl
    )
axStim.set_xlim(-.06, 1.5)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 4,
    sharey = axStim
)
for idx_bin, bin in enumerate(binCentres):
    axResp.fill_between(
        times_RespLckd, 
        gavChrsp_respLckd_binned[idx_bin] - semChresp_respLckd_binned[idx_bin], 
        gavChrsp_respLckd_binned[idx_bin] + semChresp_respLckd_binned[idx_bin],
        alpha = .3, color = cols[idx_bin]
    )
    axResp.plot(
        times_RespLckd,
        gavChrsp_respLckd_binned[idx_bin],
        color = cols[idx_bin], label = bin
    )
axResp.set_xlim(-.5, .02)

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.03, .32)
axStim.set_yticks([0, .1, .2])
axStim.set_yticklabels(['0', '.1', '.2'])
axStim.set_ylabel('Channel response [a.u.]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax == 0:
        ax.spines['bottom'].set_bounds(0, 1.5)
        ax.spines['left'].set_bounds(0, .2)
    else:
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_bounds(-.5, 0)
        ax.yaxis.set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
hndls, lbls = axStim.get_legend_handles_labels()
fig.legend(
    hndls[-3:], lbls[-3:],
    frameon = False,
    ncol = 3, columnspacing = 1,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (.5, .95),
    bbox_transform = fig.transFigure
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'gavChrsp_binned_v02.png', dpi = 600)
plt.close()
# %% computing tuning strength
for dset in good_sets:
    dataEEG[dset]['tuningStrength'] = {}
    for epochs_key, epochs_val in dataEEG[dset]['chRspSrtdSmth'].items():
         dataEEG[dset]['tuningStrength'][epochs_key] = (
             epochs_val * np.cos(channels)[..., None]
         ).sum(1)
aggTuning_stimLckd, aggTuning_respLckd = [
    np.array([
        dataEEG[dset]['tuningStrength'][epochs].mean(0)
        for dset in dataEEG]) 
    for epochs in ['stimulus-locked', 'response-locked']
]
# DONE: stim- and resp-locked analyses of tuning strength as a function of:
# response accuracy, response speed, and decision difficulty
# %% computing tuning strengths relative to response accuracy
aggTuning_ERR = [np.array([[
    dataEEG[dset]['tuningStrength'][epochs][dataEEG[dset]['bhv']['error'] == 0].mean(0),
    dataEEG[dset]['tuningStrength'][epochs][dataEEG[dset]['bhv']['error'] == 1].mean(0)
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavTuning_ERR = [epoch.mean(0) for epoch in aggTuning_ERR]
seTuning_ERR = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggTuning_ERR
]
gavTuning_ERR_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavTuning_ERR
]
seTuning_ERR_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seTuning_ERR
]

t_StimLckd, p_StimLckd = stats.ttest_1samp(
    aggTuning_stimLckd, 0
)
padj_StimLckd = multitest.fdrcorrection(
    p_StimLckd
)[1]

t_RespLckd, p_RespLckd = stats.ttest_1samp(
    aggTuning_respLckd, 0
)
padj_RespLckd = multitest.fdrcorrection(
    p_RespLckd
)[1]

t_StimLckd_ERR, p_StimLckd_ERR = stats.ttest_rel(
    aggTuning_ERR[0][:, 0], aggTuning_ERR[0][:, 1]
)
padj_StimLckd_ERR = multitest.fdrcorrection(
    p_StimLckd_ERR
)[1]

t_RespLckd_ERR, p_RespLckd_ERR = stats.ttest_rel(
    aggTuning_ERR[1][:, 0], aggTuning_ERR[1][:, 1]
)
padj_RespLckd_ERR = multitest.fdrcorrection(
    p_RespLckd_ERR
)[1]

# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Correct', 'Error']):
    axStim.fill_between(
        times_StimLckd, 
        gavTuning_ERR_smth[0][idx_cond] - seTuning_ERR_smth[0][idx_cond], 
        gavTuning_ERR_smth[0][idx_cond] + seTuning_ERR_smth[0][idx_cond],
        alpha = .3, color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavTuning_ERR_smth[0][idx_cond],
        color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)
for idx_test, name_test in enumerate(['All vs 0', 'Correct vs Error']):
    axStim_pval.plot(
        times_StimLckd,
        -np.log10([
            padj_StimLckd,
            padj_StimLckd_ERR
        ][idx_test] / .05),
        color = ln_col, 
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_test],
        label = name_test
    )
axStim_pval.axhline(0, lw = .75, color = 'gray')
# axStim_pval.set_ylim(-10, 25)
axStim_pval.set_ylim(-5.5, 17.5)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Correct', 'Error']):
    axResp.fill_between(
        times_RespLckd, 
        gavTuning_ERR_smth[1][idx_cond] - seTuning_ERR_smth[1][idx_cond], 
        gavTuning_ERR_smth[1][idx_cond] + seTuning_ERR_smth[1][idx_cond],
        alpha = .3, color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavTuning_ERR_smth[1][idx_cond],
        color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
for idx_test, name_test in enumerate(['All vs 0', 'Correct vs Error']):
    axResp_pval.plot(
        times_RespLckd,
        -np.log10([
            padj_RespLckd,
            padj_RespLckd_ERR
        ][idx_test] / .05),
        color = ln_col, 
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_test],
        label = name_test
    )
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.3, 1.3)
axStim.set_yticks([0, .5])
axStim.set_yticklabels(['0', '.5'])
axStim.set_ylabel('Tuning strength [a.u.]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (.98, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, .5)
        else:
            # ax.spines['left'].set_bounds(0, 25)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'motionTuning_ERR.png', dpi = 600)
plt.close()
# %% computing channel response relative to response accuracy
aggChResp_ERR = [np.array([[
    dataEEG[dset]['chRspSrtdSmth'][epochs][dataEEG[dset]['bhv']['error'] == 0, 7:9].mean(0).mean(0),
    dataEEG[dset]['chRspSrtdSmth'][epochs][dataEEG[dset]['bhv']['error'] == 1, 7:9].mean(0).mean(0)
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavChResp_ERR = [epoch.mean(0) for epoch in aggChResp_ERR]
seTuning_ERR = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggTuning_ERR
]
gavTuning_ERR_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavTuning_ERR
]
seTuning_ERR_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seTuning_ERR
]

t_StimLckd, p_StimLckd = stats.ttest_1samp(
    aggTuning_stimLckd, 0
)
padj_StimLckd = multitest.fdrcorrection(
    p_StimLckd
)[1]

t_RespLckd, p_RespLckd = stats.ttest_1samp(
    aggTuning_respLckd, 0
)
padj_RespLckd = multitest.fdrcorrection(
    p_RespLckd
)[1]

t_StimLckd_ERR, p_StimLckd_ERR = stats.ttest_rel(
    aggTuning_ERR[0][:, 0], aggTuning_ERR[0][:, 1]
)
padj_StimLckd_ERR = multitest.fdrcorrection(
    p_StimLckd_ERR
)[1]

t_RespLckd_ERR, p_RespLckd_ERR = stats.ttest_rel(
    aggTuning_ERR[1][:, 0], aggTuning_ERR[1][:, 1]
)
padj_RespLckd_ERR = multitest.fdrcorrection(
    p_RespLckd_ERR
)[1]

# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Correct', 'Error']):
    axStim.fill_between(
        times_StimLckd, 
        gavTuning_ERR_smth[0][idx_cond] - seTuning_ERR_smth[0][idx_cond], 
        gavTuning_ERR_smth[0][idx_cond] + seTuning_ERR_smth[0][idx_cond],
        alpha = .3, color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavTuning_ERR_smth[0][idx_cond],
        color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)
for idx_test, name_test in enumerate(['All vs 0', 'Correct vs Error']):
    axStim_pval.plot(
        times_StimLckd,
        -np.log10([
            padj_StimLckd,
            padj_StimLckd_ERR
        ][idx_test] / .05),
        color = ln_col, 
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_test],
        label = name_test
    )
axStim_pval.axhline(0, lw = .75, color = 'gray')
# axStim_pval.set_ylim(-10, 25)
axStim_pval.set_ylim(-5.5, 17.5)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Correct', 'Error']):
    axResp.fill_between(
        times_RespLckd, 
        gavTuning_ERR_smth[1][idx_cond] - seTuning_ERR_smth[1][idx_cond], 
        gavTuning_ERR_smth[1][idx_cond] + seTuning_ERR_smth[1][idx_cond],
        alpha = .3, color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavTuning_ERR_smth[1][idx_cond],
        color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
for idx_test, name_test in enumerate(['All vs 0', 'Correct vs Error']):
    axResp_pval.plot(
        times_RespLckd,
        -np.log10([
            padj_RespLckd,
            padj_RespLckd_ERR
        ][idx_test] / .05),
        color = ln_col, 
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_test],
        label = name_test
    )
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.3, 1.3)
axStim.set_yticks([0, .5])
axStim.set_yticklabels(['0', '.5'])
axStim.set_ylabel('Tuning strength [a.u.]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (.98, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, .5)
        else:
            # ax.spines['left'].set_bounds(0, 25)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'motionTuning_ERR.png', dpi = 600)
plt.close()
# %% computing tuning strength relative to response speed
aggTuning_RT = [np.array([[
    dataEEG[dset]['tuningStrength'][epochs][
        dataEEG[dset]['bhv']['error'] == 0
    ][
       pd.qcut(
           dataEEG[dset]['bhv'].loc[dataEEG[dset]['bhv']['error'] == 0, 'RT'], 
           q = 3, 
           labels = False
        ).values == 0 # first RT tercile
    ].mean(0),
    dataEEG[dset]['tuningStrength'][epochs][
        dataEEG[dset]['bhv']['error'] == 0
    ][
       pd.qcut(
           dataEEG[dset]['bhv'].loc[dataEEG[dset]['bhv']['error'] == 0, 'RT'], 
           q = 3, 
           labels = False
        ).values == 2 # third RT tercile
    ].mean(0),
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavTuning_RT = [epoch.mean(0) for epoch in aggTuning_RT]
seTuning_RT = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggTuning_RT
]
gavTuning_RT_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavTuning_RT
]
seTuning_RT_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seTuning_RT
]

t_StimLckd_RT, p_StimLckd_RT = stats.ttest_rel(
    aggTuning_RT[0][:, 0], aggTuning_RT[0][:, 1]
)
padj_StimLckd_RT = multitest.fdrcorrection(
    p_StimLckd_RT
)[1]

t_RespLckd_RT, p_RespLckd_RT = stats.ttest_rel(
    aggTuning_RT[1][:, 0], aggTuning_RT[1][:, 1]
)
padj_RespLckd_RT = multitest.fdrcorrection(
    p_RespLckd_RT
)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Fast', 'Slow']):
    axStim.fill_between(
        times_StimLckd, 
        gavTuning_RT_smth[0][idx_cond] - seTuning_RT_smth[0][idx_cond], 
        gavTuning_RT_smth[0][idx_cond] + seTuning_RT_smth[0][idx_cond],
        alpha = .3, color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavTuning_RT_smth[0][idx_cond],
        color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)

axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_StimLckd_RT / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)
axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-3.75, 11.25)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Fast', 'Slow']):
    axResp.fill_between(
        times_RespLckd, 
        gavTuning_RT_smth[1][idx_cond] - seTuning_RT_smth[1][idx_cond], 
        gavTuning_RT_smth[1][idx_cond] + seTuning_RT_smth[1][idx_cond],
        alpha = .3, color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavTuning_RT_smth[1][idx_cond],
        color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_RespLckd_RT / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.3, 1.3)
axStim.set_yticks([0, .5])
axStim.set_yticklabels(['0', '.5'])
axStim.set_ylabel('Tuning strength [a.u.]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (.98, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, .5)
        else:
            # ax.spines['left'].set_bounds(0, 25)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'motionTuning_RT.png', dpi = 600)
plt.close()
# %% computing tuning strength relative to response speed
aggChResp_RT = [np.array([[
    dataEEG[dset]['chRspSrtdSmth'][epochs][
        dataEEG[dset]['bhv']['error'] == 0, 7:9
    ].mean(1)[
       pd.qcut(
           dataEEG[dset]['bhv'].loc[dataEEG[dset]['bhv']['error'] == 0, 'RT'], 
           q = 3, 
           labels = False
        ).values == 0 # first RT tercile
    ].mean(0),
    dataEEG[dset]['chRspSrtdSmth'][epochs][
        dataEEG[dset]['bhv']['error'] == 0, 7:9
    ].mean(1)[
       pd.qcut(
           dataEEG[dset]['bhv'].loc[dataEEG[dset]['bhv']['error'] == 0, 'RT'], 
           q = 3, 
           labels = False
        ).values == 2 # third RT tercile
    ].mean(0),
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavTuning_RT = [epoch.mean(0) for epoch in aggTuning_RT]
seTuning_RT = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggTuning_RT
]
gavTuning_RT_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavTuning_RT
]
seTuning_RT_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seTuning_RT
]

t_StimLckd_RT, p_StimLckd_RT = stats.ttest_rel(
    aggTuning_RT[0][:, 0], aggTuning_RT[0][:, 1]
)
padj_StimLckd_RT = multitest.fdrcorrection(
    p_StimLckd_RT
)[1]

t_RespLckd_RT, p_RespLckd_RT = stats.ttest_rel(
    aggTuning_RT[1][:, 0], aggTuning_RT[1][:, 1]
)
padj_RespLckd_RT = multitest.fdrcorrection(
    p_RespLckd_RT
)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Fast', 'Slow']):
    axStim.fill_between(
        times_StimLckd, 
        gavTuning_RT_smth[0][idx_cond] - seTuning_RT_smth[0][idx_cond], 
        gavTuning_RT_smth[0][idx_cond] + seTuning_RT_smth[0][idx_cond],
        alpha = .3, color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavTuning_RT_smth[0][idx_cond],
        color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)

axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_StimLckd_RT / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)
axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-3.75, 11.25)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Fast', 'Slow']):
    axResp.fill_between(
        times_RespLckd, 
        gavTuning_RT_smth[1][idx_cond] - seTuning_RT_smth[1][idx_cond], 
        gavTuning_RT_smth[1][idx_cond] + seTuning_RT_smth[1][idx_cond],
        alpha = .3, color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavTuning_RT_smth[1][idx_cond],
        color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_RespLckd_RT / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.3, 1.3)
axStim.set_yticks([0, .5])
axStim.set_yticklabels(['0', '.5'])
axStim.set_ylabel('Tuning strength [a.u.]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (.98, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, .5)
        else:
            # ax.spines['left'].set_bounds(0, 25)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'motionTuning_RT.png', dpi = 600)
plt.close()
# %% computing tuning strength relative to decision difficulty
aggTuning_DIFF = [np.array([[
    dataEEG[dset]['tuningStrength'][epochs][
        (dataEEG[dset]['bhv']['error'] == 0)
        & (dataEEG[dset]['bhv']['decSig'] == 30)
    ].mean(0),
    dataEEG[dset]['tuningStrength'][epochs][
        (dataEEG[dset]['bhv']['error'] == 0)
        & (dataEEG[dset]['bhv']['decSig'] == 15)
    ].mean(0)
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavTuning_DIFF = [epoch.mean(0) for epoch in aggTuning_DIFF]
seTuning_DIFF = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggTuning_DIFF
]
gavTuning_DIFF_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavTuning_DIFF
]
seTuning_DIFF_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seTuning_DIFF
]

t_StimLckd_DIFF, p_StimLckd_DIFF = stats.ttest_rel(
    aggTuning_DIFF[0][:, 0], aggTuning_DIFF[0][:, 1]
)
padj_StimLckd_DIFF = multitest.fdrcorrection(
    p_StimLckd_DIFF
)[1]

t_RespLckd_DIFF, p_RespLckd_DIFF = stats.ttest_rel(
    aggTuning_DIFF[1][:, 0], aggTuning_DIFF[1][:, 1]
)
padj_RespLckd_DIFF = multitest.fdrcorrection(
    p_RespLckd_DIFF
)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Easy', 'Hard']):
    axStim.fill_between(
        times_StimLckd, 
        gavTuning_DIFF_smth[0][idx_cond] - seTuning_DIFF_smth[0][idx_cond], 
        gavTuning_DIFF_smth[0][idx_cond] + seTuning_DIFF_smth[0][idx_cond],
        alpha = .3, color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavTuning_DIFF_smth[0][idx_cond],
        color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)

axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_StimLckd_DIFF / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Easy vs Hard'
)
axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-.417, 1.25)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Easy', 'Hard']):
    axResp.fill_between(
        times_RespLckd, 
        gavTuning_DIFF_smth[1][idx_cond] - seTuning_DIFF_smth[1][idx_cond], 
        gavTuning_DIFF_smth[1][idx_cond] + seTuning_DIFF_smth[1][idx_cond],
        alpha = .3, color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavTuning_DIFF_smth[1][idx_cond],
        color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_RespLckd_DIFF / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Easy vs Hard'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.3, 1.3)
axStim.set_yticks([0, .5])
axStim.set_yticklabels(['0', '.5'])
axStim.set_ylabel('Tuning strength [a.u.]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (1, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (.98, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, .5)
        else:
            # ax.spines['left'].set_bounds(0, 25)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'motionTuning_DIFF.png', dpi = 600)
plt.close()
# %% DONE: compute peaks at tunning curves
# there are no shifts in tuning curves relative to criterion orientation
aggTunCurves = dict([
    (dset, dataEEG[dset]['chRspSrtdSmth']['response-locked'].mean(-1))
    for dset in dataEEG
])
# flipping channels for trials in which the motion was CCW relative to criterion
for dset in aggTunCurves:
    idx_flip = dataEEG[dset]['bhv']['decAlt'].values < 0
    aggTunCurves[dset][idx_flip] = aggTunCurves[dset][idx_flip][:, ::-1]

avTunCurves_ERR = np.array([
    [
        aggTunCurves[dset][dataEEG[dset]['bhv']['error'] == error].mean(0)
        for error in [0, 1]
    ]
    for dset in aggTunCurves
])
#===========================================================================
# %% EMS ANALYSES
#===========================================================================
times_StimLckd= np.linspace(-.250, 2, int(2.250*256))
times_RespLckd = np.linspace(-.500, .020, int(.520*256))
# temporal smoothing parameters
SD = int(.016 * 256)
gwins = {}
for idx_epochs, duration_epochs in enumerate([2.250, .520]):
    gwin = scipy.signal.gaussian(int(duration_epochs * 256), SD)
    gwin /= gwin.sum()
    gwins[
        ['stimulus-locked', 'response-locked'][idx_epochs]
    ] = gwin
# loading filtered trials
for dset in good_sets:
    logging.warning('Loading filtered trials for {}'.format(dset))
    fname = savePath + dset + '_emsFilters.hdf'
    dataEEG[dset]['emsFilter'] = {}
    with h5py.File(fname, 'r') as f:
        for key_name, key_value in f.items():
            fltrdTrls = f[key_name]['filteredTrials'][:]
            # smoothing channel responses
            fltrdTrlsSmth = scipy.signal.convolve(
                fltrdTrls,
                gwins[key_name][None],
                mode = 'same'
            )
            dataEEG[dset]['emsFilter'][key_name] = fltrdTrlsSmth
# DONE: stim- and resp-locked analyses of decision decoding as a function of:
# response accuracy, response speed, and decision difficulty
aggDecision_stimLckd, aggDecision_respLckd = [np.array([
    dataEEG[dset]['emsFilter'][epochs].mean(0) for dset in dataEEG
]) for epochs in ['stimulus-locked', 'response-locked']]
# %% response accuracy
aggDecision_ERR = [np.array([[
    dataEEG[dset]['emsFilter'][epochs][dataEEG[dset]['bhv']['error'] == 0].mean(0),
    dataEEG[dset]['emsFilter'][epochs][dataEEG[dset]['bhv']['error'] == 1].mean(0)
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavDecision_ERR = [epoch.mean(0) for epoch in aggDecision_ERR]
seDecision_ERR = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggDecision_ERR
]
gavDecision_ERR_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavDecision_ERR
]
seDecision_ERR_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seDecision_ERR
]

t_StimLckd, p_StimLckd = stats.ttest_1samp(
    aggDecision_stimLckd, 0
)
padj_StimLckd = multitest.fdrcorrection(
    p_StimLckd
)[1]

t_RespLckd, p_RespLckd = stats.ttest_1samp(
    aggDecision_respLckd, 0
)
padj_RespLckd = multitest.fdrcorrection(
    p_RespLckd
)[1]

t_StimLckd_ERR, p_StimLckd_ERR = stats.ttest_rel(
    aggDecision_ERR[0][:, 0], aggDecision_ERR[0][:, 1]
)
padj_StimLckd_ERR = multitest.fdrcorrection(
    p_StimLckd_ERR
)[1]

t_RespLckd_ERR, p_RespLckd_ERR = stats.ttest_rel(
    aggDecision_ERR[1][:, 0], aggDecision_ERR[1][:, 1]
)
padj_RespLckd_ERR = multitest.fdrcorrection(
    p_RespLckd_ERR
)[1]

# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Correct', 'Error']):
    axStim.fill_between(
        times_StimLckd, 
        gavDecision_ERR_smth[0][idx_cond] - seDecision_ERR_smth[0][idx_cond], 
        gavDecision_ERR_smth[0][idx_cond] + seDecision_ERR_smth[0][idx_cond],
        alpha = .3, color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavDecision_ERR_smth[0][idx_cond],
        color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)
for idx_test, name_test in enumerate(['All vs 0', 'Correct vs Error']):
    axStim_pval.plot(
        times_StimLckd,
        -np.log10([
            padj_StimLckd,
            padj_StimLckd_ERR
        ][idx_test] / .05),
        color = ln_col, 
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_test],
        label = name_test
    )
axStim_pval.axhline(0, lw = .75, color = 'gray')
# axStim_pval.set_ylim(-2.5, 6.25)
axStim_pval.set_ylim(-1.92, 5.75)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Correct', 'Error']):
    axResp.fill_between(
        times_RespLckd, 
        gavDecision_ERR_smth[1][idx_cond] - seDecision_ERR_smth[1][idx_cond], 
        gavDecision_ERR_smth[1][idx_cond] + seDecision_ERR_smth[1][idx_cond],
        alpha = .3, color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavDecision_ERR_smth[1][idx_cond],
        color = colpal[['Pacific Blue', 'Heliotrope'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
for idx_test, name_test in enumerate(['All vs 0', 'Correct vs Error']):
    axResp_pval.plot(
        times_RespLckd,
        -np.log10([
            padj_RespLckd,
            padj_RespLckd_ERR
        ][idx_test] / .05),
        color = ln_col, 
        dashes = [
            (1, 0),
            (1, .5)
        ][idx_test],
        label = name_test
    )
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.35, .35)
axStim.set_yticks([-.2, 0, .2])
axStim.set_yticklabels(['-.2', '0', '.2'])
axStim.set_ylabel('Decision decoding [z]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (.98, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(-.2, .2)
        else:
            # ax.spines['left'].set_bounds(0, 25)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'decisionDecoding_ERR.png', dpi = 600)
plt.close()
# %% computing tuning strength relative to response speed
aggDecision_RT = [np.array([[
    dataEEG[dset]['emsFilter'][epochs][
        dataEEG[dset]['bhv']['error'] == 0
    ][
       pd.qcut(
           dataEEG[dset]['bhv'].loc[dataEEG[dset]['bhv']['error'] == 0, 'RT'], 
           q = 3, 
           labels = False
        ).values == 0 # first RT tercile
    ].mean(0),
    dataEEG[dset]['emsFilter'][epochs][
        dataEEG[dset]['bhv']['error'] == 0
    ][
       pd.qcut(
           dataEEG[dset]['bhv'].loc[dataEEG[dset]['bhv']['error'] == 0, 'RT'], 
           q = 3, 
           labels = False
        ).values == 2 # third RT tercile
    ].mean(0),
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavDecision_RT = [epoch.mean(0) for epoch in aggDecision_RT]
seDecision_RT = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggDecision_RT
]
gavDecision_RT_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavDecision_RT
]
seDecision_RT_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seDecision_RT
]

t_StimLckd_RT, p_StimLckd_RT = stats.ttest_rel(
    aggDecision_RT[0][:, 0], aggDecision_RT[0][:, 1]
)
padj_StimLckd_RT = multitest.fdrcorrection(
    p_StimLckd_RT
)[1]

t_RespLckd_RT, p_RespLckd_RT = stats.ttest_rel(
    aggDecision_RT[1][:, 0], aggDecision_RT[1][:, 1]
)
padj_RespLckd_RT = multitest.fdrcorrection(
    p_RespLckd_RT
)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Fast', 'Slow']):
    axStim.fill_between(
        times_StimLckd, 
        gavDecision_RT_smth[0][idx_cond] - seDecision_RT_smth[0][idx_cond], 
        gavDecision_RT_smth[0][idx_cond] + seDecision_RT_smth[0][idx_cond],
        alpha = .3, color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavDecision_RT_smth[0][idx_cond],
        color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)

axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_StimLckd_RT / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)
axStim_pval.axhline(0, lw = .75, color = 'gray')
# axStim_pval.set_ylim(-2.5, 6.25)
axStim_pval.set_ylim(-1.25, 3.75)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Fast', 'Slow']):
    axResp.fill_between(
        times_RespLckd, 
        gavDecision_RT_smth[1][idx_cond] - seDecision_RT_smth[1][idx_cond], 
        gavDecision_RT_smth[1][idx_cond] + seDecision_RT_smth[1][idx_cond],
        alpha = .3, color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavDecision_RT_smth[1][idx_cond],
        color = colpal[['CG Blue', 'Medium Turquoise'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_RespLckd_RT / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Fast vs Slow'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.05, .35)
axStim.set_yticks([0, .2])
axStim.set_yticklabels(['0', '.2'])
axStim.set_ylabel('Decision decoding [z]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (.98, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (1, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0,.2)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'decisionDecoding_RT.png', dpi = 600)
plt.close()
# %% computing tuning strength relative to decision difficulty
aggDecision_DIFF = [np.array([[
    dataEEG[dset]['emsFilter'][epochs][
        (dataEEG[dset]['bhv']['error'] == 0)
        & (dataEEG[dset]['bhv']['decSig'] == 30)
    ].mean(0),
    dataEEG[dset]['emsFilter'][epochs][
        (dataEEG[dset]['bhv']['error'] == 0)
        & (dataEEG[dset]['bhv']['decSig'] == 15)
    ].mean(0)
] for dset in dataEEG]) for epochs in ['stimulus-locked', 'response-locked']]

gavDecision_DIFF = [epoch.mean(0) for epoch in aggDecision_DIFF]
seDecision_DIFF = [
    np.sqrt(
        np.var(
            epoch 
            # removing participant's intercept
            - epoch.mean(1)[:, None] 
            # adding overall intercept
            + epoch.mean(0).mean(0)[None, None],
            axis = 0,
            ddof = 1
        ) 
        # correcting for within-measures underestimate of variance
        * (epoch.shape[1] / (epoch.shape[1] - 1)) 
        # normalizing by sample size for SEM
        / epoch.shape[0]
    ) 
    for epoch in aggDecision_DIFF
]
gavDecision_DIFF_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in gavDecision_DIFF
]
seDecision_DIFF_smth = [
    mne.filter.filter_data(
        epoch, 256, None, 10, method = 'iir'
    )
    for epoch in seDecision_DIFF
]

t_StimLckd_DIFF, p_StimLckd_DIFF = stats.ttest_rel(
    aggDecision_DIFF[0][:, 0], aggDecision_DIFF[0][:, 1]
)
padj_StimLckd_DIFF = multitest.fdrcorrection(
    p_StimLckd_DIFF
)[1]

t_RespLckd_DIFF, p_RespLckd_DIFF = stats.ttest_rel(
    aggDecision_DIFF[1][:, 0], aggDecision_DIFF[1][:, 1]
)
padj_RespLckd_DIFF = multitest.fdrcorrection(
    p_RespLckd_DIFF
)[1]
# %% plotting
sbn.set_style('ticks')
ln_col = '#3F3F3F' # dark gray
colpal = {
    "Heliotrope":"#d883ff",
    "Vivid Violet":"#9336fd",
    "Magenta Crayola":"#f050ae",
    "Light Coral":"#f77976",
    "Maize Crayola":"#f9c74f",
    "Pistachio":"#90be6d",
    "Zomp":"#43aa8b",
    "Cadet Blue":"#4d908e",
    "Queen Blue":"#577590",
    "CG Blue":"#277da1",
    "Pacific Blue":"#33a8c7",
    "Medium Turquoise":"#52e3e1",
}

# figsize = (2.7, 1.35)
figsize = (3.10, 1.56)
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : SSIZE,
          'ytick.labelsize' : SSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : SSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : SSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)

fig = plt.figure(figsize=figsize)

axStim = plt.subplot2grid(
    (4, 4), (0, 0), colspan = 3, rowspan = 3
)
for idx_cond, name_cond in enumerate(['Easy', 'Hard']):
    axStim.fill_between(
        times_StimLckd, 
        gavDecision_DIFF_smth[0][idx_cond] - seDecision_DIFF_smth[0][idx_cond], 
        gavDecision_DIFF_smth[0][idx_cond] + seDecision_DIFF_smth[0][idx_cond],
        alpha = .3, color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]]
    )
    axStim.plot(
        times_StimLckd,
        gavDecision_DIFF_smth[0][idx_cond],
        color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]],
        label = name_cond
    )
axStim.set_xlim(-.06, 1.5)

axStim_pval = plt.subplot2grid(
    (4, 4), (3, 0), colspan = 3,
    sharex = axStim,
)

axStim_pval.plot(
    times_StimLckd,
    -np.log10(padj_StimLckd_DIFF / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Easy vs Hard'
)
axStim_pval.axhline(0, lw = .75, color = 'gray')
axStim_pval.set_ylim(-1.25, 3.75)

axResp = plt.subplot2grid(
    (4, 4), (0, 3), rowspan = 3,
    sharey = axStim
)
for idx_cond, name_cond in enumerate(['Easy', 'Hard']):
    axResp.fill_between(
        times_RespLckd, 
        gavDecision_DIFF_smth[1][idx_cond] - seDecision_DIFF_smth[1][idx_cond], 
        gavDecision_DIFF_smth[1][idx_cond] + seDecision_DIFF_smth[1][idx_cond],
        alpha = .3, color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]]
    )
    axResp.plot(
        times_RespLckd,
        gavDecision_DIFF_smth[1][idx_cond],
        color = colpal[['Maize Crayola', 'Pistachio'][idx_cond]],
        label = name_cond
    )
axResp.set_xlim(-.5, .02)

axResp_pval = plt.subplot2grid(
    (4, 4), (3, 3),
    sharex = axResp, sharey = axStim_pval
)
axResp_pval.plot(
    times_RespLckd,
    -np.log10(padj_RespLckd_DIFF / .05),
    color = ln_col, 
    dashes = (1, 0),
    label = 'Easy vs Hard'
)
axResp_pval.axhline(0, lw = .75, color = 'gray')

axStim.set_title('Stimulus-locked')
axStim.set_ylim(-.05, .35)
axStim.set_yticks([0, .2])
axStim.set_yticklabels(['0', '.2'])
axStim.set_ylabel('Decision decoding [z]')
axStim.set_xticks([0, .5, 1, 1.5])
axStim.legend(
    *axStim.get_legend_handles_labels(),
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 9, bbox_to_anchor = (1, 1)
)
axStim.set_xticklabels(['0 s', '.5', '1', '1.5'])

axStim_pval.set_ylabel(r'-log$_{10}(\frac{p_{FDR}}{.05})$', rotation = 0)
axStim_pval.legend(
    *axStim_pval.get_legend_handles_labels(),
    ncol = 2, columnspacing = 1,
    frameon = False,
    handlelength = 1, handletextpad = .5,
    loc = 4, bbox_to_anchor = (.98, 1)
)
axResp.set_title('Response-locked')
axResp.set_xticks([-.5, 0])
axResp.set_xticklabels(['-.5', '0 s'])

for idx_ax, ax in enumerate(fig.axes):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if idx_ax in [0, 1]:
        ax.spines['bottom'].set_bounds(0, 1.5)
        if idx_ax == 0:
            ax.spines['left'].set_bounds(0, .2)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
    else:
        ax.spines['bottom'].set_bounds(-.5, 0)
    if idx_ax in [2, 3]:
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
    if idx_ax in [0, 2]:
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)

fig.tight_layout(
    rect = (.01, 0, .99, 1), pad = 0
)
plt.subplots_adjust(hspace = .675, wspace = .5)
fig.savefig(exportPath + 'decisionDecoding_DIFF.png', dpi = 600)
plt.close()
# %%
