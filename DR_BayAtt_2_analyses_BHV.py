'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2020-09-17
-----
Last Modified: 2020-09-17
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2020 Dragan Rangelov, The University of Queensland
'''
# DONE: Re-run analyses with EEG outliers excluded
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
# %% importing libraries
#===============================================================================
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
import sys, os
sys.path.insert(0, scriptPath)
# os.environ['R_HOME'] = '/home/ubuntu/anaconda3/envs/dev/lib/R'
# %load_ext rpy2.ipython
# os.environ['PYTHONIOENCODING'] = 'UTF-8'
import mmodel as mm
import numpy as np
import seaborn as sbn
sbn.set()
import pandas as pd
import pickle
import h5py
from collections import OrderedDict
from statsmodels.stats.anova import AnovaRM
from statsmodels import stats
from scipy import stats as spss
# #===============================================================================
# # Compiling behavioural data
# #===============================================================================
# # loading original files
# bhvFiles = sorted([f for f in os.listdir(rawPath) if 'main_data' in f])
# bhvData = OrderedDict()
# for bhvfile in bhvFiles:
#     tmp = pd.read_csv(rawPath + bhvfile, sep='\t')
#     dset = bhvfile.split('_')[0]
#     if dset in ['s04', 's17', 's20']:
#         tmp = tmp[np.where(tmp['runningTrialNo'].values == 0)[0][-1]:]
#     bhvData[dset] = tmp
# # loading log files
# logFiles = sorted([f for f in os.listdir(rawPath) if 'main_log' in f])
# logData = OrderedDict()
# for logfile in logFiles:
#     tmp = pd.read_csv(rawPath + logfile, sep='\t', names=['time', 'mssg', 'log'])
#     dset = logfile.split('_')[0]
#     # removing the incorrectly logged data for s04 and s17 (see notes)
#     if dset in ['s04', 's17', 's20']:
#         tmp = tmp[np.where(tmp['log'].values == 'START_BLOCK_1')[0][-1]:]
#     # where in the log file the trial starts
#     if dset != 's22':
#         idxtrl = np.array([np.where(tmp['log'] == log)[0]
#                            for log in tmp['log']
#                            if '_TRIAL_' in log]).squeeze()
#     else:
#         idxtrl = np.array([np.where(tmp['log'] == log)[0]
#                            for log in tmp['log']
#                            if '_TRIAL_' in log and 'BLOCK_16' not in log]).squeeze()
#     # how many trials have we logged
#     goodTrials = np.zeros(idxtrl.shape[0])
#     # which of these trials are the good trials
#     goodtrl = (tmp['log'].values[idxtrl + 3] == 'SIGNAL_ON').squeeze()
#     goodTrials[goodtrl] = 1
#     bhvData[dset]['goodTrials'] = goodTrials
# for dset in bhvData:
#     bhvData[dset].to_csv(bhvPath + dset + '_bhv_data.txt', sep = '\t', index = False)
#===============================================================================
# %% Loading behavioural data
#===============================================================================
dsets = [
    dfile[:-8] 
    for dfile in os.listdir(dataPath) 
    if dfile not in ['.DS_Store','._.DS_Store']
]
dropsets = ['s09'] # dropping dsets which were not recorded correctly
dsets = sorted(filter(lambda x: x not in dropsets, dsets))
bad_sets = ['s01','s10','s25']
good_sets = [dset for dset in dsets if dset not in bad_sets]

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
    dataEEG[dset]['badTiming'] = dropdFrames
    dataEEG[dset]['missedRsp'] = missedResp
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
#===============================================================================
# %% aggregating BHV data
#===============================================================================
aggBHV = pd.concat([dset['bhv'] for dset in dataEEG.values()])
# %% computing quartiles
avRT = aggBHV.groupby([
    'Subject Number',
    'error',
    'decSig'
])['RT'].apply(np.quantile, q = [.25, .50, .75]).reset_index().rename(
    columns = {'Subject Number': 'sNo'}
)
avRT = pd.concat([
    avRT,
    pd.DataFrame(
        data = np.vstack(avRT['RT'].values),
        columns = ['.25', '.50', '.75']
    )
], axis = 1).drop(columns = ['RT'])
avRT = avRT.melt(
    id_vars = ['sNo', 'error', 'decSig'],
    value_vars = ['.25', '.50', '.75'],
    var_name = 'quartile',
    value_name = 'RT'
)
# correcting RTs for participant-specific intercept
# avRT = avRT.merge(
#     avRT.groupby(['sNo', 'error', 'quartile'])['RT'].mean().reset_index(),
#     on = ['sNo', 'error', 'quartile'],
#     suffixes = ['', '_errQuartPerSub'],
# )
# avRT = avRT.merge(
#     avRT.groupby(['error','quartile'])['RT'].mean().reset_index(),
#     on = ['error', 'quartile'],
#     suffixes = ['', '_errQuart'],
# )
avRT = avRT.merge(
    avRT.groupby(['sNo'])['RT'].mean().reset_index(),
    on = ['sNo'],
    suffixes = ['', '_sub'],
)
avRT['RT_tot'] = avRT['RT'].mean()
# # normalized RTs per quartile
# avRT['zRT_errQuart'] = avRT['RT'] - avRT['RT_errQuartPerSub'] + avRT['RT_errQuart']
# normalized RTs for the full design (quartile and dCrit)
avRT['zRT_tot'] = avRT['RT'] - avRT['RT_sub'] + avRT['RT_tot']
gavRT = pd.merge(
    avRT.groupby(['error', 'decSig', 'quartile'])['RT'].mean().reset_index().rename(
        columns = {'RT': 'gavRT'}
    ),
    avRT.groupby(['error', 'decSig', 'quartile'])['zRT_tot'].apply(
        lambda x: np.sqrt(np.var(x, ddof = 1) * (12 / 11)) / np.sqrt(len(x))
    ).reset_index().rename(columns = {'zRT_tot': 'seRT'}),
    on = ['error', 'decSig', 'quartile']
)
gavRT.groupby(['error'])['gavRT'].mean()
# computing error rates
avERR = aggBHV.groupby([
    'Subject Number', 
    'decSig'
])['error'].mean().reset_index().rename(
    columns = {'Subject Number' : 'sNo'}
)
avERR = avERR.merge(
    avERR.groupby(['sNo'])['error'].mean().reset_index(),
    on = ['sNo'],
    suffixes = ['', '_sub'],
)
avERR['error_tot'] = avERR['error'].mean()
# normalized ERR rate
avERR['zERR_tot'] = avERR['error'] - avERR['error_sub'] + avERR['error_tot']
gavERR = pd.merge(
    avERR.groupby(['decSig'])['error'].mean().reset_index().rename(
        columns = {'error': 'gavERR'}
    ),
    avERR.groupby(['decSig'])['zERR_tot'].apply(
        lambda x: np.sqrt(np.var(x, ddof = 1) * (2 / 1)) / np.sqrt(len(x))
    ).reset_index().rename(columns = {'zERR_tot': 'seERR'}),
    on = ['decSig']
)
# %% analysing RT
for col in ['sNo', 'error', 'decSig', 'quartile']:
    avRT[col + '_fac'] = avRT[col].astype('str')
# %%
%%R -i avRT
aov_RT <- aov(
    RT ~ error_fac + decSig_fac * quartile_fac + Error(sNo_fac / (error_fac * decSig_fac * quartile_fac)),
    data = avRT
)
summary(aov_RT)
# %% pairwise comparisons
effD_corr = avRT.loc[
    avRT['error'] == 0
].groupby('quartile').apply(
    lambda x: spss.ttest_rel(
        x.loc[x['decSig'] == 15, 'RT'],
        x.loc[x['decSig'] == 30, 'RT']
    )[1]
).reset_index().rename(columns = {0: 'p_val'})
effD_corr['p_adj'] = stats.multitest.fdrcorrection(effD_corr['p_val'])[1]
effD_err = avRT.loc[
    avRT['error'] == 1
].groupby('quartile').apply(
    lambda x: spss.ttest_rel(
        x.loc[x['decSig'] == 15, 'RT'],
        x.loc[x['decSig'] == 30, 'RT']
    )[1]
).reset_index().rename(columns = {0: 'p_val'})
effD_err['p_adj'] = stats.multitest.fdrcorrection(effD_err['p_val'])[1]
# %% analysing errors
for col in ['sNo', 'decSig']:
    avERR[col + '_fac'] = avERR[col].astype('str')
# %%
%%R -i avERR
aov_ERR <- aov(
    error ~ decSig_fac + Error(sNo_fac / decSig_fac),
    data = avERR
)
summary(aov_ERR)
#===============================================================================
# %% plotting BHV data
#===============================================================================
sbn.set_style('ticks')
cols = np.array(sbn.husl_palette(h = .5, s = .75))[[1, 3]]
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
ln_col = '#3F3F3F' # dark gray
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

axERR = plt.subplot2grid((1, 10), (0, 0), colspan = 3)
for idx_dcrit, dcrit in enumerate([15, 30]):
    ssGavERR = gavERR.loc[gavERR['decSig'] == dcrit]
    axERR.bar(
        np.array([.125, -.125])[idx_dcrit],
        ssGavERR['gavERR'],
        width = .325,
        color = colpal[['Pistachio','Maize Crayola'][idx_dcrit]],
        label = ['Hard', 'Easy decision'][idx_dcrit],
        linewidth = .5,
        yerr = ssGavERR['seERR'],
        error_kw = dict(
            elinewidth = .75
        )
    )
# for lineY in avERR[
#     ['sNo', 'decSig', 'zERR_tot']
# ].pivot(
#     index = 'sNo',
#     columns = 'decSig',
#     values = 'zERR_tot'
# ).reset_index()[[15, 30]].values:
#     axERR.plot(
#         np.array([.125, -.125]),
#         lineY,
#         color = ln_col,
#         lw = .7,
#         alpha = .3
#     )
axERR.set_xlim(-.5, .5)
axERR.set_xticks([])
axERR.set_ylabel('Error rate [%]')
axERR.set_ylim(0, .175)
axERR.set_yticks(np.linspace(0, .15, 4))
axERR.set_yticklabels(['0', '5', '10', '15'])
for spine in ['top', 'right', 'bottom']:
    axERR.spines[spine].set_visible(False)
axERR.spines['left'].set_bounds(0, .15)

axRT = plt.subplot2grid((1, 10), (0, 4), colspan = 6)

for idx_dcrit, dcrit in enumerate([15, 30]):
    ssGavRT = gavRT.loc[(gavRT['error'] == 0) & (gavRT['decSig'] == dcrit)]
    axRT.bar(
        np.arange(3) * .65 + np.array([.125, -.125])[idx_dcrit],
        ssGavRT['gavRT'],
        width = .325,
        color = colpal[['Pistachio','Maize Crayola'][idx_dcrit]],
        label = ['Hard', 'Easy decision'][idx_dcrit],
        linewidth = .5,
        yerr = ssGavRT['seRT'],
        error_kw = dict(
            elinewidth = .75
        )
    )
# for idx_q, q in enumerate(avRT['quartile'].unique()):
#     ssRT = avRT.loc[
#         (avRT['error'] == 0)
#         & (avRT['quartile'] == q), 
#         ['sNo', 'decSig', 'zRT_errQuart']
#     ].pivot(
#         index = 'sNo',
#         columns = 'decSig',
#         values = 'zRT_errQuart'
#     )
#     for lineY in ssRT[[15, 30]].values:
#         axRT.plot(
#             (idx_q * .65 + np.array([.125, -.125])),
#             lineY,
#             color = ln_col,
#             lw = .7,
#             alpha = .3
#         )

for idx_dcrit, dcrit in enumerate([15, 30]):
    ssGavRT = gavRT.loc[(gavRT['error'] == 1) & (gavRT['decSig'] == dcrit)]
    axRT.bar(
        (np.arange(3) * .65 + np.array([.125, -.125])[idx_dcrit]) + 2.5,
        ssGavRT['gavRT'],
        width = .325,
        color = colpal[['Pistachio','Maize Crayola'][idx_dcrit]],
        label = ['Hard', 'Easy decision'][idx_dcrit],
        linewidth = .5,
        yerr = ssGavRT['seRT'],
        error_kw = dict(
            elinewidth = .75
        )
    )
# for idx_q, q in enumerate(avRT['quartile'].unique()):
#     ssRT = avRT.loc[
#         (avRT['error'] == 1)
#         & (avRT['quartile'] == q), 
#         ['sNo', 'decSig', 'zRT_errQuart']
#     ].pivot(
#         index = 'sNo',
#         columns = 'decSig',
#         values = 'zRT_errQuart'
#     )
#     for lineY in ssRT[[15, 30]].values:
#         axRT.plot(
#             (idx_q * .65 + np.array([.125, -.125])) + 2.5,
#             lineY,
#             color = ln_col,
#             lw = .7,
#             alpha = .3
#         )

axRT.set_ylabel('RT percentile [ms]')
axRT.set_ylim(.300, 1.000)
axRT.set_yticks(np.linspace(.300, .900, 4))
axRT.set_yticklabels(['300', '500', '700', '900'])
for spine in ['top', 'right', 'bottom']:
    axRT.spines[spine].set_visible(False)
axRT.tick_params('x', length = 0, pad = 2)
axRT.set_xticks(np.concatenate([
    (np.arange(3) * .65) + offset
    for offset in [0, 2.5]
]))
axRT.set_xticklabels(
    [r'25$^{th}$', r'50$^{th}$', r'75$^{th}$'] * 2,
    fontsize = 8
)
axRT.spines['left'].set_bounds(.3, .9)
axRT.text(
    .65, .9, 
    'Correct', fontsize = SSIZE,
    ha = 'center', va = 'top', 
    transform = axRT.transData
)
axRT.text(
    .65 + 2.5, .9, 
    'Error', fontsize = SSIZE,
    ha = 'center', va = 'top', 
    transform = axRT.transData
)

hndls, lbls = axERR.get_legend_handles_labels()
axERR.legend(
    hndls[::-1], 
    lbls[::-1],
    handlelength = 1, handletextpad = .5,
    frameon = False,
    ncol = 2, columnspacing = 1,
    loc = 9,
    bbox_to_anchor = [.5, 1.05],
    bbox_transform = fig.transFigure,
    title_fontsize = 8
)

plt.subplots_adjust(
    top = .92, bottom = .11, left = .12, right = 1,
    hspace = 0, wspace = .25
)
fig.savefig(exportPath + 'gavBHV_withErrRT_withSE.png', dpi = 600)
plt.close()
# %%
