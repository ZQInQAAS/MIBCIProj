import os
import numpy as np
from utils import eemd, bsscca_ifc
from offline import data_4class
from scipy.io import savemat


def find_path(path, subject, date_file, idx):
    subject_path = os.path.join(path, subject)
    date_file_path = os.path.join(subject_path, date_file)
    npz_or_txt = os.listdir(date_file_path)
    npz_list = [npz for npz in npz_or_txt if npz.endswith('.npz')]
    data_path = os.path.join(date_file_path, npz_list[idx])
    return data_path


def ICA_EEMD_CCA(data_path):
    # return (samples, channels)
    data = data_4class(data_path)
    data.removeEOGbyICA()
    data = data.get_raw_data()
    data_eemdcca = np.zeros(data.shape)
    for i in range(data.shape[1]):
    # for i in range(1):
        raw_eemd = eemd(np.squeeze(data[:, i]))
        raw_cca = bsscca_ifc(raw_eemd, remove_idx=[0, -2, -1])
        data_eemdcca[:, i] = np.sum(raw_cca, 0)
    return data_eemdcca


def offline_one(path, subject, date_file=None):
    subject_path = os.path.join(path, subject)
    dir_or_files = os.listdir(subject_path)
    for i in range(1 if date_file else len(dir_or_files)-1):
        date_file_path = os.path.join(subject_path, date_file if date_file else dir_or_files[i])
        npz_or_txt = os.listdir(date_file_path)
        npz_list = [npz for npz in npz_or_txt if npz.endswith('.npz')]
        # for j in range(len(npz_list)-1):
        for j in range(1, 2):
            train_path = os.path.join(date_file_path, npz_list[j])
            data_eemdcca = ICA_EEMD_CCA(train_path)
            print(dir_or_files[i] + 'finished. shape:' + str(data_eemdcca.shape))
            # matpath = dir_or_files[i] + 'data.mat'
            matpath = os.path.join(subject_path, date_file) + 'data.mat'
            savemat(matpath, {'data': data_eemdcca})


if __name__ == '__main__':
    path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large'
    data_path = offline_one(path,  'lichenyang', 'lichenyang_20190616')

