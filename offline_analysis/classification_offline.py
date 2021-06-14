import os
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from MIdataset_NF import MIdataset, cal_power_feature
from process_tools import sliding_win, csp_filter
from sklearn.model_selection import ShuffleSplit, cross_val_score


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
    # data = data_4class(path)
    data = MIdataset(path)
    data.bandpass_filter(8, 30)
    # data.removeEOGbyICA()
    x_epoch, y_epoch = data.get_epoch_data(select_label=label)
    x_epoch, y_epoch = sliding_win(x_epoch, y_epoch, step=500, window=500)
    return x_epoch, y_epoch


def classification(train_x, train_y, test_x, test_y):
    csp = csp_filter(m=3)
    clf = svm.SVC(C=0.8, kernel='rbf')
    tmp_train = csp.fit_transform(train_x, train_y)
    clf.fit(tmp_train, train_y)
    after_csp_test_x = csp.transform(test_x)
    predict = clf.predict(after_csp_test_x)
    right_sum = np.sum(predict == test_y)
    Acc = right_sum / len(test_y)
    # kappa = Kappa(test_y, predict, Acc)
    kappa = 0
    return Acc, kappa


def tenfold(path, label):
    # 单个文件十折交叉验证
    data_x, data_y = preprocess(path, label)
    np.delete(data_x, [32, 42], 1)  # M1 M2
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    cv_split = cv.split(data_y)
    Acc_list, kappa_list = [], []
    for train_idx, test_idx in cv_split:
        train_x, test_x = data_x[:, :, train_idx], data_x[:, :, test_idx]
        train_y, test_y = data_y[train_idx], data_y[test_idx]
        acc, kappa = classification(train_x, train_y, test_x, test_y)
        Acc_list.append(acc)
        # kappa_list.append(kappa)
    # Acc = round(np.mean(Acc_list), 2)
    Acc = np.mean(Acc_list)
    # kappa = round(np.mean(kappa_list), 2)
    return Acc, 0


def acc(path, la):
    # la = ['Right',‘Left’, 'Rest']
    accl = []
    for i in range(5):
        acc, k = tenfold(path, la)
        accl.append(acc)
    print('Acc is ', round(np.mean(accl), 3))
    return np.mean(accl)


if __name__ == '__main__':
    p = os.path.abspath(os.path.dirname(os.getcwd()))
    p = p + r'\data_set\XY\XY_20210611'
    path = p + r'\Acq_pre_20210611_1555_16.npz'
    path2 = p + r'\Acq_pre_20210611_1602_27.npz'
    path3 = p + r'\Acq_pre_20210611_1609_52.npz'
    path4 = p + r'\Acq_post_20210611_1649_16.npz'
    path5 = p + r'\Acq_post_20210611_1659_33.npz'
    path6 = p + r'\Acq_post_20210611_1708_21.npz'
    la = ['Right', 'Rest']
    acc(path, la)
    acc(path2, la)
    acc(path3, la)
    acc(path4, la)
    acc(path5, la)
    acc(path6, la)