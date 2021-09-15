import os
import re
import numpy as np
import pandas as pd
from MIdataset_NF import MIdataset
from BCIConfig import events_id_3mi, event_id_mrt, ch_names60, fs, ch_types60, pick_rest_ch


def match(pattern, filename_list):
    # 正则匹配文件名  pattern：'Acq'
    matchlist = []
    for i in filename_list:
        if re.search(pattern, i):
            matchlist.append(i)
    matchlist.sort()
    # print(matchlist)
    return matchlist


def rela_power_pip(dataset_path, subject_set):
    df_power = pd.DataFrame(columns=['Subject_name', 'Subject_id', 'Run'] + ch_names60)
    df_rela_power = pd.DataFrame(columns=['Subject_name', 'Subject_id', 'Run'] + ch_names60)
    notEEG_list = ['log', 'model.npz', 'MRT_pre_answer.npz', 'MRT_post_answer.npz', 'Baseline_pre.npz',
                   'Baseline_post.npz', 'prebaseline', 'other']
    label = 'Right'  # Acq: 'Right', 'Left', 'Rest' /MRT: 'StartOfMR' / Bas or Res: '1'
    pattern = 'Acq'  # 'Acq'6run or 'MRT'2run  'Bas'4run 'Res'6run
    # Acq 6 run: 3 post + 3 pre 时间顺序
    # MRT 2 run: 1 post + 1 pre
    # Bas 4 run: post_ec, post_eo, pre_ec, pre_eo
    # Res 6 run: NFT 时间顺序
    Bas_run_list = ['post_ec', 'post_eo', 'pre_ec', 'pre_eo']
    Acq_run_list = ['post1', 'post2', 'post3', 'pre1', 'pre2', 'pre3']
    for s_idx in range(len(subject_set)):
        s = subject_set[s_idx]
        IAF = subject_paf[s_idx]
        # theta(4, 8), alpha(8, 13), beta(13, 30), low gamma(30, 50)=>
        # delta = (1, 3.5), theta =(IAF-6, IAF-2), alpha = (IAF-2, IAF+2),
        # beta1 = (IAF+2, IAF+8), beta2 = (IAF+8,IAF+14), beta3 = (IAF+14, 30) Corsi, M. C,2020
        # beta1 = (IAF+2, IAF+11), beta2 = (IAF+11,IAF+20), low gamma = (30, 50) Escolano, C.,2014
        fmin, fmax = IAF-2, IAF+2
        tmin, tmax = 0, 4  # Acq: (-4, 0) (0, 4) MRT: (-3, 0) (1, 4) Bas/Res (0,2)
        sub_path = os.path.join(dataset_path, s)
        date_files = os.listdir(sub_path)
        date_file_path = os.path.join(sub_path, date_files[0])
        npz_files = os.listdir(date_file_path)
        mi_file_list = match(pattern, npz_files)  # 'MRT'
        mi_file_list.sort()
        k = 0
        for j in range(len(mi_file_list)):
            if mi_file_list[j] not in notEEG_list:
                k = k + 1
                if pattern == 'Acq':
                    power_s = [s, s_idx + 1, Acq_run_list[k-1], ]
                    rela_power_s = [s, s_idx + 1, Acq_run_list[k-1], ]
                elif pattern == 'Bas':
                    power_s = [s, s_idx + 1, Bas_run_list[k - 1], ]
                    rela_power_s = [s, s_idx + 1, Bas_run_list[k - 1], ]
                else:
                    power_s = [s, s_idx + 1, k, ]
                    rela_power_s = [s, s_idx + 1, k, ]
                path = os.path.join(date_file_path, mi_file_list[j])
                power, rela_power = get_power_from_epoch(path, pattern, fmin, fmax, tmin, tmax, label)  # MRT/Acq
                # power, rela_power = get_rawdata_power(path, fmin, fmax, pattern)  # Res/Bas
                power_s.extend(power.tolist())
                rela_power_s.extend(rela_power.tolist())
                df_power.loc[len(df_power)] = power_s
                df_rela_power.loc[len(df_rela_power)] = rela_power_s
        print('finish classification.', s)
    df_power.to_csv('df_power_Acq_Right_alpha.csv', index=False)
    df_rela_power.to_csv('df_rela_power_Acq_Right_alpha.csv', index=False)

def get_rawdata_power(path, fmin, fmax, pattern):
    # Res/Bas 不切epoch Bas 60s Res 180s
    eo_baseline_pre = dict(np.load(path, allow_pickle=True))
    baseline_eo = eo_baseline_pre['signal']
    if pattern == 'Res':
        Rest_start = eo_baseline_pre['events'][0][0]
        baseline_eo = baseline_eo[Rest_start:, :]
    # start, end = 2500, 27505  # baseline  50s
    # data = baseline_eo[start:end, :]
    power, rela_power = MI.get_relapower_byarray(baseline_eo.T, ch=ch_names60, fmin=fmin, fmax=fmax, basefmin=1)
    return power, rela_power

def get_power_from_epoch(path, pattern, fmin, fmax, tmin, tmax, label):
    # MRT/Acq
    epochs_mne = read_clean_data(path, pattern) if is_clean else read_raw_data(path, pattern)
    power, rela_power = MI.get_rela_power(epochs_mne, fmin, fmax, basefmin=1, basefmax=50,
                      tmin=tmin, tmax=tmax, label=label)
    return power, rela_power

def read_clean_data(path, pattern):
    epochs_mne = MI.read_cleandata_path(path)
    if pattern in ['Res', 'Bas']:
        epochs_mne = get_epoch(MI, epochs_mne)  # 10s切成2s的epoch
    return epochs_mne

def read_raw_data(path, pattern):
    data, events = MI.read_rawdata_path(path)
    raw_mne = MI.get_raw_mne(data.T)
    raw_mne = MI.drop_channel(raw_mne, ch=['M1', 'M2', 'HEOG', 'VEOG'])
    if pattern in ['MRT', 'Acq']:
        eventid = event_id_mrt if pattern == 'MRT' else events_id_3mi
        epochs_mne = MI.get_epochs_mne(raw_mne, events=events, event_id=eventid, tmin=-5, tmax=4)
    else:  # 'Res', 'Bas' 切 epoch
        tmin, tmax = 0, 2
        if pattern == 'Bas':  # Baseline
            n_epochs, st = 25, 5  # st开始时间
        else:  # Rest alpha feedback
            n_epochs, st = 17, 14.5
        start, stop = st * fs, (st + n_epochs * tmax) * fs + n_epochs
        raw_array = raw_mne.get_data(start=int(start), stop=int(stop))  # (n_ch, n_times)
        epoch_array = raw_array.reshape(60, n_epochs, -1)  # (n_ch, n_epochs, n_times)
        epoch_array = epoch_array.swapaxes(0, 1)  # (n_epochs, n_ch, n_times)
        epochs_mne = MI.get_epochs_mne_byarray(epoch_array, ch_name=ch_names60, ch_type=ch_types60, tmin=tmin)
    return epochs_mne

def get_epoch(MI, epochs_mne):
    epoch_array = epochs_mne.get_data()  # (n_epochs, n_ch, n_times)
    epoch_array = epoch_array[:, :, :5000].transpose(1, 2, 0)  # return (n_ch, n_times, n_epochs)
    epoch_array = epoch_array.reshape(60, 1000, -1)  # 10s切成2s的epoch 2s*500=1000
    epoch_array = epoch_array.transpose(2, 0, 1)  # return (n_epochs, n_ch, n_times)
    epochs_mne = MI.get_epochs_mne_byarray(epoch_array, ch_name=epochs_mne.ch_names, ch_type='eeg', tmin=0)
    return epochs_mne

def cal_power():
    path = r'D:\Myfiles\MIBCI_NF\data_set\CYJ\CYJ_20210703\Acq_pre_20210703_1603_38.npz'  # CYJ pre2 F4 power有问题
    IAF = 10
    label = 'Right'
    fmin, fmax = IAF - 2, IAF + 2
    tmin, tmax = 0, 4
    power, rela_power = get_power_from_epoch(path, 'Acq', fmin, fmax, tmin, tmax, label)
    # data, events = MI.read_rawdata_path(path)
    # raw_mne = MI.get_raw_mne(data.T)
    # raw_mne = MI.drop_channel(raw_mne, ch=['M1', 'M2'])
    # MI.plot_raw_psd(raw_mne)
    print(1)

if __name__ == '__main__':
    dataset_path = r'D:\Myfiles\MIBCI_NF\data_set'
    # dataset_path = r'D:\Myfiles\MIBCI_NF\data_set_clean'
    is_clean = False
    subject_set = ['PNM', 'XY', 'CYH', 'WRQ', 'ZXY', 'YZT', 'WXH', 'LXT', 'FX', 'SXF', 'WCC',
                   'HYD', 'XW', 'WYQ', 'CQY', 'LY', 'MYH', 'MHJ',  'LYR', 'WY', 'CYJ', 'CZ']  # 22 subject
    subject_paf = [9.5, 10, 10, 10, 11, 10.25, 10.5, 10.5, 10.25, 10.25, 9.75,
                   9.75, 9, 9.5, 10.5, 10, 9.75, 9.25, 9.5, 10.5, 10, 10.5]
    MI = MIdataset()
    # rela_power_pip(dataset_path, subject_set)
    cal_power()