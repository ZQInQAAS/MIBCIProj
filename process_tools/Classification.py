import numpy as np
from sklearn import svm
from process_tools import csp_filter, bandpass_filter


class Classification(object):
    def __init__(self):
        self.csp = csp_filter(m=3)
        self.filter_low = 8
        self.filter_high = 30
        self.svm_clf = svm.SVC(C=0.8, kernel='rbf')

    def train_model(self, train_x, train_y, fs):
        # train_x: (sample, channal, trial)
        train_x = np.delete(train_x, [11, 12], 1)  # 移除f4 cp3
        x_train_filt = bandpass_filter(train_x, fs, self.filter_low, self.filter_high)
        # x_train_filt, train_y = sliding_window(x_train_filt, train_y)
        tmp_train = self.csp.fit_transform(x_train_filt, train_y)
        self.svm_clf.fit(tmp_train, train_y)

    def online_predict(self, epoch, fs):
        # epoch : T×N  单个epoch T: 采样点数  N: 通道数
        # predict: ndarray(1,) double 分类结果
        epoch = np.delete(epoch, [11, 12], 1)  # 移除f4 cp3
        after_filter_test_x = bandpass_filter(epoch, fs, self.filter_low, self.filter_high)
        after_csp_test_x = self.csp.transform(after_filter_test_x)
        predict = self.svm_clf.predict(after_csp_test_x)
        return predict
