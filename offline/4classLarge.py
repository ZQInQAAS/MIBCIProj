import os
import xlwt
import random
import numpy as np
from sklearn import svm
from offline import data_4class
from utils import CSP, sliding_window


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


def mne_preprocess(path):
    data = data_4class(path)
    data.bandpass_filter(8, 30)
    x_epoch, y_epoch = data.get_epoch_data()
    x_epoch, y_epoch = sliding_window(x_epoch, y_epoch)
    return x_epoch, y_epoch


def offline_test(train_path, test_path):
    train_x, train_y = mne_preprocess(train_path)
    test_x, test_y = mne_preprocess(test_path)
    csp = CSP(m=3)
    clf = svm.SVC(C=0.8, kernel='rbf')
    tmp_train = csp.fit_transform(train_x, train_y)
    clf.fit(tmp_train, train_y)
    after_csp_test_x = csp.transform(test_x)
    predict = clf.predict(after_csp_test_x)
    right_sum = np.sum(predict == test_y)
    Acc = round(right_sum / len(test_y), 2)
    kappa = round(Kappa(test_y, predict, Acc), 2)
    return Acc, kappa


def offline_all(path):
    subject_set = ('zhujunjie', 'lichenyang', 'guanhaonan', 'sunmingjian',
                   'renyizuo', 'zhouqing', 'jiangyingyan', 'zhengqi')
    f = xlwt.Workbook()
    for subject in subject_set:
        sheet = f.add_sheet(subject, cell_overwrite_ok=True)
        sub_path = os.path.join(path, subject)
        dir_or_files = os.listdir(sub_path)
        for i in range(len(dir_or_files)-1):
            date_file_path = os.path.join(sub_path, dir_or_files[i])
            npz_or_txt = os.listdir(date_file_path)
            npz_list = [npz for npz in npz_or_txt if npz.endswith('.npz')]
            for j in range(len(npz_list) - 1):
                train_path = os.path.join(date_file_path, npz_list[j])
                test_path = os.path.join(date_file_path, npz_list[j+1])
                acc, kappa = offline_test(train_path, test_path)
                print(dir_or_files[i] + ' run.' + str(j+2) + ':' + str(acc))
                sheet.write(j + i * 5, 0, acc, )
                sheet.write(j + i * 5, 2, kappa)
    f.save('4class_all.xls')


def offline_one(path, subject, date_file=None):
    f = xlwt.Workbook()
    sheet = f.add_sheet(subject)
    subject_path = os.path.join(path, subject)
    dir_or_files = os.listdir(subject_path)
    for i in range(1 if date_file else len(dir_or_files)-1):
        date_file_path = os.path.join(subject_path, date_file if date_file else dir_or_files[i])
        npz_or_txt = os.listdir(date_file_path)
        npz_list = [npz for npz in npz_or_txt if npz.endswith('.npz')]
        for j in range(len(npz_list)-1):
            train_path = os.path.join(date_file_path, npz_list[j])
            test_path = os.path.join(date_file_path, npz_list[j+1])
            acc, kappa = offline_test(train_path, test_path)
            print(dir_or_files[i] + ' run.' + str(j+2) + ':' + str(acc))
            sheet.write(j+i*5, 0, acc)
            sheet.write(j+i*5, 2, kappa)
    f.save(subject + '_4class.xls')


if __name__ == '__main__':
    path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large'
    offline_one(path, 'guanhaonan', 'guanhaonan_20190620')
    # offline_all(path)
