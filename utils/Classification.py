import numpy as np
from sklearn import svm
from utils.loadrawnpz import sliding_window
from utils import csp_train, csp_spatial_filter, bandpass_filter


class Classification(object):
    def __init__(self):
        self.m = 3
        self.filter_low = 8
        self.filter_high = 30
        self.csp_proj_matrix = None
        self.svm_clf = None

    def train_model(self, train_x, train_y, fs):
        x_train_filt = bandpass_filter(train_x, fs, self.filter_low, self.filter_high)
        x_train_filt, train_y = sliding_window(x_train_filt, train_y)
        self.csp_proj_matrix = csp_train(x_train_filt, train_y, self.m)
        tmp_train = csp_spatial_filter(x_train_filt, self.csp_proj_matrix)
        self.svm_clf = svm.SVC(C=0.8, kernel='rbf')
        self.svm_clf.fit(tmp_train, train_y)

    def online_predict(self, epoch, fs):
        # epoch : T×N  单个epoch T: 采样点数  N: 通道数
        # predict: ndarray(1,) double 分类结果
        # epoch = CARFilter(epoch)
        epoch = np.delete(epoch, [11, 12], 1)  # 移除f4 cp3
        after_filter_test_x = bandpass_filter(epoch, fs, self.filter_low, self.filter_high)
        after_csp_test_x = csp_spatial_filter(after_filter_test_x, self.csp_proj_matrix)
        predict = self.svm_clf.predict(after_csp_test_x)
        return predict