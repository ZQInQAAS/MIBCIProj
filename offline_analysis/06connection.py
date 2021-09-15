import os
import re
import numpy as np
import pandas as pd
import pingouin as pg
from BCIConfig import fs, ch_names60, ch_types60
from MIdataset_NF import MIdataset
from process_tools import pvalue_3D
from mne.viz import plot_sensors_connectivity


def match(pattern, filename_list):
    # 正则匹配文件名  pattern：'Acq'
    matchlist = []
    for i in filename_list:
        if re.search(pattern, i):
            matchlist.append(i)
    matchlist.sort()
    # print(matchlist)
    return matchlist


def get_array(path):
    # Res/Bas 不切epoch Bas 60s Res 180s
    eo_baseline_pre = dict(np.load(path, allow_pickle=True))
    baseline_eo = eo_baseline_pre['signal']
    if pattern == 'Res':
        Rest_start = baseline_eo['events'][0][0]
        baseline_eo = baseline_eo[Rest_start:, :]
    return baseline_eo


def get_epoch(path):
    data, events = MI.read_rawdata_path(path)
    raw_mne = MI.get_raw_mne(data.T)
    raw_mne = MI.drop_channel(raw_mne, ch=['M1', 'M2', 'HEOG', 'VEOG'])
    # 'Res', 'Bas' 切 epoch
    tmin, tmax = 0, 2
    if pattern == 'Bas':  # Baseline 60s
        n_epochs, st = 28, 1  # st开始时间
    else:  # Rest alpha feedback 180s
        n_epochs, st = 17, 14.5
    start, stop = st * fs, (st + n_epochs * tmax) * fs + n_epochs
    raw_array = raw_mne.get_data(start=int(start), stop=int(stop))  # (n_ch, n_times)
    epoch_array = raw_array.reshape(60, n_epochs, -1)  # (n_ch, n_epochs, n_times)
    epoch_array = epoch_array.swapaxes(0, 1)  # (n_epochs, n_ch, n_times)
    epochs_mne = MI.get_epochs_mne_byarray(epoch_array, ch_name=ch_names60, ch_type=ch_types60, tmin=tmin)
    return epochs_mne


def save_conv_pip():
    ch = 60
    n_run = 4
    con_s = np.zeros([len(subject_set), n_run, ch, ch])
    for s_idx in range(len(subject_set)):
        s = subject_set[s_idx]
        IAF = subject_paf[s_idx]
        fmin, fmax = IAF - 2, IAF + 2
        # delta = (1, IAF-6), theta =(IAF-6, IAF-2), alpha = (IAF-2, IAF+2),
        # beta11 = (IAF+2, IAF+8), beta12 = (IAF+8,IAF+14), beta13 = (IAF+14, 30)
        # beta21 = (IAF+2, IAF+11), beta22 = (IAF+11,30), low gamma = (30, 50)
        sub_path = os.path.join(dataset_path, s)
        date_files = os.listdir(sub_path)
        date_file_path = os.path.join(sub_path, date_files[0])
        npz_files = os.listdir(date_file_path)
        mi_file_list = match(pattern, npz_files)
        mi_file_list.sort()
        con_run = np.zeros([len(mi_file_list), ch, ch])
        k = -1
        for j in range(len(mi_file_list)):
            if mi_file_list[j] not in notEEG_list:
                k = k + 1
                npz_file_path = os.path.join(date_file_path, mi_file_list[j])
                epochs = get_epoch(npz_file_path)
                con = MI.get_spectral_connectivity(epochs, fmin, fmax, tmin=0, method='pli')
                con_run[k, :, :] = con
        con_s[s_idx, :, :, :] = con_run[:k+1, :, :]
    np.savez('../analysis_df/conv/pli_con_matrix_alpha.npz', con_s=con_s)


def load_conv(path):
    npz_data_alpha = np.load(path)
    con_matrix = np.abs(dict(npz_data_alpha)['con_s'])  # run 'post_ec', 'post_eo', 'pre_ec', 'pre_eo'
    con_matrix_preeo = con_matrix[:, 3, :, :]
    con_matrix_posteo = con_matrix[:, 1, :, :]
    return con_matrix_preeo, con_matrix_posteo


def plot_conv():
    conv_pre, conv_post = load_conv(r'D:\Myfiles\MIBCI_NF\analysis_df\conv\pli_con_matrix_alpha.npz')
    con_matrix_premeansub = np.mean(conv_pre, axis=0)
    con_matrix_postmeansub = np.mean(conv_post, axis=0)
    change = con_matrix_postmeansub - con_matrix_premeansub
    epochs = get_epoch(r'D:\Myfiles\MIBCI_NF\data_set\PNM\PNM_20210607\Baseline_post_eo.npz')
    # plot_sensors_connectivity(epochs.info, con_matrix_premeansub)
    # plot_sensors_connectivity(epochs.info, con_matrix_postmeansub)


def plot_conv_p():
    conv_pre, conv_post = load_conv(r'D:\Myfiles\MIBCI_NF\analysis_df\conv\pli_con_matrix_alpha.npz')
    p = pvalue_3D(conv_pre, conv_post, ispaired=True)
    epochs = get_epoch(r'D:\Myfiles\MIBCI_NF\data_set\PNM\PNM_20210607\Baseline_post_eo.npz')
    plot_sensors_connectivity(epochs.info, p)


if __name__ == '__main__':
    dataset_path = r'D:\Myfiles\MIBCI_NF\data_set'
    notEEG_list = ['log', 'model.npz', 'MRT_pre_answer.npz', 'MRT_post_answer.npz', 'Baseline_pre.npz',
                   'Baseline_post.npz', 'prebaseline', 'other']
    subject_set = ['PNM', 'XY', 'CYH', 'WRQ', 'ZXY', 'YZT', 'WXH', 'LXT', 'FX', 'SXF', 'WCC',
                   'HYD', 'XW', 'WYQ', 'CQY', 'LY', 'MYH', 'MHJ', 'LYR', 'WY', 'CYJ', 'CZ']  # 22 subject
    subject_paf = [9.5, 10, 10, 10, 11, 10.25, 10.5, 10.5, 10.25, 10.25, 9.75,
                   9.75, 9, 9.5, 10.5, 10, 9.75, 9.25, 9.5, 10.5, 10, 10.5]
    is_clean = False
    pattern = 'Bas'
    MI = MIdataset()
    # save_conv_pip()
    # plot_conv()
    plot_conv_p()
    print('Finished.')
