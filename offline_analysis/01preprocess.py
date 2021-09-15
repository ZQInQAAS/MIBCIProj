import os
import numpy as np
import matlab.engine  # 仅支持python36以下
import pandas as pd
import pickle
from BCIConfig import fs, events_id_3mi, event_id_mrt, ch_types60, ch_names60, ch_types62, ch_names62
from MIdataset_NF import MIdataset
from scipy.io import savemat


def preprocess(npz_file_path, file_type):
    # data = data_4class(npz_file_path)
    MI = MIdataset()
    data, events = MI.read_rawdata_path(npz_file_path)
    raw_mne = MI.get_raw_mne(data.T)
    raw_mne = MI.drop_channel(raw_mne, ch=['M1', 'M2'])
    # raw_mne = MI.bandpass_filter(raw_mne, 1, 100)
    # raw_mne = MI.set_reference(raw_mne)  # CAR
    # eng = matlab.engine.start_matlab()
    # path = os.path.dirname(os.getcwd()) + r'\MIBCIOffline\process_tools'
    path = r'D:\Myfiles\codes\ReMAE\ReMAEfunc'  #  path = r'../ReMAEfunc/'
    # eng.cd(path, nargout=0)
    # epoch
    if file_type in ['Acq', 'MRT']:
        if file_type == 'Acq':
            eventid = events_id_3mi
            tmin, tmax = -5, 4
        else:
            eventid = event_id_mrt
            tmin, tmax = -3, 8
        epochs_mne = MI.get_epochs_mne(raw_mne, events=events, event_id=eventid, tmin=tmin, tmax=tmax)
    elif file_type in ['Bas', 'Res']:
        tmin, tmax = 0, 10
        # events, eventid = None, None
        if file_type == 'Bas':  # Baseline
            n_epochs, st = 5, 5  # st开始时间
        else:  # Rest alpha feedback
            n_epochs, st = 17, 14.5
        start, stop = st * fs, (st + n_epochs * tmax) * fs + n_epochs
        raw_array = raw_mne.get_data(start=int(start), stop=int(stop))  # (n_ch, n_times)
        epoch_array = raw_array.reshape(62, n_epochs, -1)  # (n_ch, n_epochs, n_times)
        epoch_array = epoch_array.swapaxes(0, 1)  # (n_epochs, n_ch, n_times)
        epochs_mne = MI.get_epochs_mne_byarray(epoch_array, ch_name=ch_names62, ch_type=ch_types62, tmin=tmin)
    else:
        return
    # epochs_mne = MI.removeEOGbyICA(epochs_mne)  # ICA
    epochs_mne = MI.drop_channel(epochs_mne, ch=['HEOG', 'VEOG'])
    # epoch_array, label = MI.get_epoch_array(epochs_mne, tmin=tmin, tmax=tmax)  # epoch_array (n_epochs, n_ch, n_times)
    # savemat('epoch_arrayLXT.mat', {'epoch_array': epoch_array})
    # epoch_mat = matlab.double(epoch_array.tolist())
    # epoch_mat_clean = eng.eemdcca_pip(epoch_mat, fs)
    # epoch_clean = np.array(epoch_mat_clean)
    # epochs_mne = MI.get_epochs_mne_byarray(epoch_clean, ch_name=ch_names60, ch_type=ch_types60,
    #                                        events=events, event_id=eventid, tmin=tmin)
    return epochs_mne


def preprocess_pip():
    dataset_path = r'D:\Myfiles\MIBCI_NF\data_set'
    subject_set = ['PNM', 'XY', 'CYH', 'WRQ', 'ZXY', 'WXH', 'FX', 'WCC',
                   'HYD', 'XW', 'WYQ', 'LY', 'MYH', 'MHJ', 'WY', 'CYJ',
                   'YZT', 'LXT', 'SXF', 'CQY', 'LYR']  # 21 subject
    notEEG_list = ['log', 'model.npz', 'MRT_pre_answer.npz', 'MRT_post_answer.npz', 'Baseline_pre.npz',
                   'Baseline_post.npz', 'prebaseline', 'other']
    for s_idx in range(len(subject_set)):
        s = subject_set[s_idx]
        sub_path = os.path.join(dataset_path, s)
        date_files = os.listdir(sub_path)
        for i in range(len(date_files)):
            date_file_path = os.path.join(sub_path, date_files[i])
            npz_files = os.listdir(date_file_path)
            for j in range(len(npz_files)):
                if npz_files[j] not in notEEG_list:
                    file_type = npz_files[j][0:3]
                    npz_file_path = os.path.join(date_file_path, npz_files[j])
                    npz_file_path = r'D:\Myfiles\MIBCI_NF\data_set\CZ\CZ_20210716\Acq_pre_20210716_1020_45.npz'
                    file_type = 'Acq'
                    epochs_mne = preprocess(npz_file_path, file_type)
                    data_clean_path = npz_file_path.replace('data_set', 'data_set_clean')
                    data_clean_path = data_clean_path.replace('npz', 'pkl')
                    if not os.path.exists(os.path.dirname(data_clean_path)):
                        os.makedirs(os.path.dirname(data_clean_path))
                    # np.savez(data_clean_path, epochs_mne=epochs_mne)
                    with open(data_clean_path, 'wb') as f:
                        f.write(pickle.dumps(epochs_mne))
                    print('finish clean data.', data_clean_path)


if __name__ == '__main__':
    preprocess_pip()
    # clean_path = r'D:\Myfiles\MIBCI_NF\data_set_clean\PNM2\PNM_20210607\Baseline_post_ec.pkl'
    # MI = MIdataset()
    # data = MI.read_cleandata_path(clean_path)
