import os
import re
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from BCIConfig import events_id_3mi, pick_motor_ch, ch_names
from MIdataset_NF import MIdataset
from process_tools import slidingwin, csp_filter
from sklearn.model_selection import ShuffleSplit, cross_val_score
from scipy.io import savemat
from sklearn.metrics import f1_score


def Kappa(test_y, predict, Acc):
    # 计算kappa值
    test_size = len(test_y)
    A1 = np.sum(test_y == 1)
    A2 = np.sum(test_y == 2)
    A3 = np.sum(test_y == 3)
    A4 = test_size - A1 - A2 - A3
    B1 = np.sum(predict == 1)
    B2 = np.sum(predict == 2)
    B3 = np.sum(predict == 3)
    B4 = test_size - B1 - B2 - B3
    Pe = (A1 * B1 + A2 * B2 + A3 * B3 + A4 * B4) / test_size**2
    kappa = (Acc - Pe) / (1 - Pe)
    return kappa


def preprocess(path, label):
    MI = MIdataset()
    if is_clean:
        if isinstance(path, list):
            epochs_mne = MI.gather_epoch(path)
        else:
            epochs_mne = MI.read_cleandata_path(path)
        epochs_mne = MI.pick_channel(epochs_mne, ch=pick_motor_ch)
        epochs_mne = MI.bandpass_filter(epochs_mne, 8, 30)
    else:
        # data = data_4class(path)
        if isinstance(path, list):
            data, events = MI.gather_data(path)
        else:
            data, events = MI.read_rawdata_path(path)
        # savemat('Post3.mat', {'data': data, 'events': events, 'ch_names': ch_names, 'fs': 500, 'events_id': events_id_3mi})
        raw_mne = MI.get_raw_mne(data.T)
        raw_mne = MI.drop_channel(raw_mne, ch=['M1', 'M2'])
        # raw_mne = MI.pick_channel(raw_mne, ch=pick_motor_ch)  # 30ch
        # raw_mne = MI.set_reference(raw_mne)  # CAR acc下降
        raw_mne = MI.bandpass_filter(raw_mne, 8, 30)
        epochs_mne = MI.get_epochs_mne(raw_mne, events=events, event_id=events_id_3mi, tmin=-5, tmax=4)
    epoch_array, label = MI.get_epoch_array(epochs_mne, tmin=0, tmax=4, select_label=label)
    epoch_array = epoch_array.swapaxes(0, 2)  # (n_epochs, n_ch, n_times)=>(n_times, n_ch, n_epochs)
    x_epoch, y_epoch = epoch_array, label
    # x_epoch, y_epoch = slidingwin(epoch_array, label, step=500, window=500, is_shuffle=False)
        # savemat('Post1_afterpreprocess.mat', {'x_epoch': x_epoch, 'y_epoch': y_epoch})
    x_epoch = np.delete(x_epoch, [23, 31, 32, 40], axis=1)  # 移除T7, T8, TP7, TP8
    return x_epoch, y_epoch


def classification(train_x, train_y, test_x, test_y):
    csp = csp_filter(m=3)
    clf = svm.SVC(C=1, kernel='rbf')  # OVR
    tmp_train = csp.fit_transform(train_x, train_y)
    clf.fit(tmp_train, train_y)
    after_csp_test_x = csp.transform(test_x)
    # savemat('Post1_aftercsp_fold4.mat', {'after_csp_train_x': tmp_train, 'train_y': train_y, 'after_csp_test_x': after_csp_test_x, 'test_y': test_y})
    predict = clf.predict(after_csp_test_x)
    right_sum = np.sum(predict == test_y)
    Acc = right_sum / len(test_y)
    # kappa = Kappa(test_y, predict, Acc)
    macro_f1 = f1_score(test_y, predict, average='macro')
    return Acc, macro_f1


def tenfold(path, label):
    # 单个文件十折交叉验证
    data_x, data_y = preprocess(path, label)
    # np.delete(data_x, [32, 42], 1)  # M1 M2
    cv = ShuffleSplit(10, test_size=0.1, random_state=42)
    cv_split = cv.split(data_y)
    Acc_list, macro_f1_list = [], []
    for train_idx, test_idx in cv_split:
        train_x, test_x = data_x[:, :, train_idx], data_x[:, :, test_idx]
        train_y, test_y = data_y[train_idx], data_y[test_idx]
        acc, macro_f1 = classification(train_x, train_y, test_x, test_y)
        Acc_list.append(acc)
        macro_f1_list.append(macro_f1)
    print(round(np.mean(Acc_list), 3))
    return np.mean(Acc_list), np.mean(macro_f1_list)


def acc_repeat(path, label, repeat=10):
    # la = ['Right',‘Left’, 'Rest']
    acclist, macro_f1_list = [], []
    for i in range(repeat):
        acc, macro_f1 = tenfold(path, label)
        acclist.append(acc)
        macro_f1_list.append(macro_f1)
    print('Acc is ', round(np.mean(acclist), 3))
    return np.mean(acclist), np.mean(macro_f1_list)


def match(pattern, filename_list):
    # 正则匹配文件名  pattern：'Acq'
    matchlist = []
    for i in filename_list:
        if re.search(pattern, i):
            matchlist.append(i)
    return matchlist


def classification_pip():
    subject_set = ['PNM2', 'XY', 'CYH', 'WRQ', 'ZXY', 'YZT', 'WXH', 'LXT', 'FX', 'SXF', 'WCC',
                   'HYD', 'XW', 'WYQ', 'CQY', 'LY', 'MYH', 'MHJ',  'LYR', 'WY', 'CYJ', 'CZ']  # 22 subject
    subject_set = ['PNM_eog',]
    subject_set = ['PNM', ]
    df_session = pd.DataFrame(columns=['Subject_name', 'Subject_id', 'Pre1', 'Pre2', 'Pre3',
                                       'Post1', 'Post2', 'Post3', 'Pre', 'Post'])
    df_session_f = pd.DataFrame(columns=['Subject_name', 'Subject_id', 'Pre1', 'Pre2', 'Pre3',
                                         'Post1', 'Post2', 'Post3', 'Pre', 'Post'])
    label = ['Right', 'Left', 'Rest']
    for s_idx in range(len(subject_set)):
        s = subject_set[s_idx]
        sub_path = os.path.join(dataset_path, s)
        date_files = os.listdir(sub_path)
        # for i in range(len(date_files)):
        acc_oneday, macrof_oneday = [], []
        date_file_path = os.path.join(sub_path, date_files[0])
        npz_files = os.listdir(date_file_path)
        mi_file_list = match('Acq', npz_files)
        mi_file_list.sort()  # pre:3 4 5  post:0 1 2
        for j in range(len(mi_file_list)):
            npz_file_path = os.path.join(date_file_path, mi_file_list[j])
            acc, macro_f = acc_repeat(npz_file_path, label, repeat=20)
            # acc, macro_f = 1, 1
            acc_oneday.append(acc)
            macrof_oneday.append(macro_f)
        # acc = [np.mean(x) for x in zip(*acc_oneday)]
        # macro_f = [np.mean(x) for x in zip(*macrof_oneday)]
        pre_path_list = [os.path.join(date_file_path, i) for i in match('Acq_post', mi_file_list)]
        post_path_list = [os.path.join(date_file_path, i) for i in match('Acq_post', mi_file_list)]
        accpre, macro_fpre = acc_repeat(pre_path_list, label, repeat=20)
        accpost, macro_fpost = acc_repeat(post_path_list, label, repeat=20)
        # accpre, macro_fpre = 1, 1
        # accpost, macro_fpost = 1, 1
        df_session = df_session.append({'Subject_name': s, 'Subject_id': s_idx + 1,
                                        'Pre1': acc_oneday[3], 'Pre2': acc_oneday[4], 'Pre3': acc_oneday[5],
                                        'Post1': acc_oneday[0], 'Post2': acc_oneday[1], 'Post3': acc_oneday[2],
                                        'Pre': accpre, 'Post': accpost}, ignore_index=True)
        df_session_f = df_session_f.append({'Subject_name': s, 'Subject_id': s_idx + 1,
                                        'Pre1': macrof_oneday[3], 'Pre2': macrof_oneday[4], 'Pre3': macrof_oneday[5],
                                        'Post1': macrof_oneday[0], 'Post2': macrof_oneday[1], 'Post3': macrof_oneday[2],
                                        'Pre': macro_fpre, 'Post': macro_fpost}, ignore_index=True)
        print('finish classification.', s)
    df_session.to_csv('df_class123_tenfold_Acc_60ch.csv', index=False)
    df_session_f.to_csv('df_class123_tenfold_MacroF_60ch.csv', index=False)


def one_pipe():
    p = os.path.abspath(os.path.dirname(os.getcwd()))
    p = p + r'\data_set\CZ\CZ_20210716\\'
    p_type = r'.npz'
    path1 = p + r'Acq_pre_20210716_1020_45' + p_type
    path2 = p + r'Acq_pre_20210716_1031_26' + p_type
    path3 = p + r'Acq_pre_20210716_1037_59' + p_type
    # path4 = p + r'Acq_post_20210716_1130_46' + p_type
    # path5 = p + r'Acq_post_20210716_1136_50' + p_type
    # path6 = p + r'Acq_post_20210716_1143_35' + p_type
    pathpre_list = [path1, path2, path3]
    # pathpost_list = [path4, path5, path6]
    la = ['Right', 'Left', 'Rest']
    tenfold(path1, la)
    tenfold(path2, la)
    tenfold(path3, la)
    # tenfold(path4, la)
    # tenfold(path5, la)
    # tenfold(path6, la)
    tenfold(pathpre_list, la)
    # tenfold(pathpost_list, la)


if __name__ == '__main__':
    # dataset_path = r'D:\Myfiles\MIBCI_NF\data_set_clean'
    # is_clean = False
    dataset_path = r'D:\Myfiles\MIBCI_NF\data_set'
    is_clean = False  # 0.63 0.778
    classification_pip()
    # one_pipe()
    print('1')
