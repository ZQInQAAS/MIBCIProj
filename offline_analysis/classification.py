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

def preprocess(path, label):
    # data = data_4class(path)
    data = MIdataset(path)
    data.bandpass_filter(8, 30)
    # data.removeEOGbyICA()
    x_epoch, y_epoch = data.get_epoch_data(select_label=label)
    x_epoch, y_epoch = sliding_win(x_epoch, y_epoch, step=500, window=500)
    return x_epoch, y_epoch

def tenfold(path, label):
    # 单个文件十折交叉验证
    data_x, data_y = preprocess(path, label)
    np.delete(data_x, [28, 29, 45], 1)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(data_y)
    Acc_list, kappa_list = [], []
    for train_idx, test_idx in cv_split:
        train_x, test_x = data_x[:, :, train_idx], data_x[:, :, test_idx]
        train_y, test_y = data_y[train_idx], data_y[test_idx]
        acc, kappa = classification(train_x, train_y, test_x, test_y)
        Acc_list.append(acc)
        kappa_list.append(kappa)
    # Acc = round(np.mean(Acc_list), 2)
    Acc = np.mean(Acc_list)
    kappa = round(np.mean(kappa_list), 2)
    return Acc, kappa

def acc(path, la):
    accl = []
    for i in range(10):
        acc, k = tenfold(path, la)
        accl.append(acc)
    print('Acc is ', round(np.mean(accl), 3))