# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as la


class csp_filter(object):
    def __init__(self, m=3):
        self.m = m
        self.csp_proj_matrix = None

    def fit(self, train_x, train_y):
        """
        训练CSP模型，生成投影矩阵F

        输入参数
        ----------
        train_x: T×N×L ndarray(或单个trial T×N)
                 T: 采样点数  N: 通道数  B: 滤波频带数  L: 训练数据trial总数
        train_y: 1 维 L 个
                L个trial对应的标签
             m: int 提取的特征对数
        返回值
        ----------
        csp_proj_matrix: 2m×N(2类csp)/2m×N×C(多类csp)
                 投影矩阵
        """
        channel_num = train_x.shape[1]
        class_list = np.sort(np.unique(train_y))
        class_num = len(class_list)
        R_one, R_rest, R_sum = self._cov_matrix(train_x, train_y, class_list)
        if class_num == 2:
            [w0, w1] = la.eig(R_one[:, :, 0], R_sum)
            I = np.argsort(-np.real(w0))
            I = (np.hstack((I[0:self.m], I[channel_num - self.m:channel_num]))).tolist()
            self.csp_proj_matrix = np.transpose(w1[:, I])
        else:
            self.csp_proj_matrix = np.zeros([2 * self.m, channel_num, class_num])
            for i in range(class_num):
                [w0, w1] = la.eig(R_one[:, :, i], R_rest[:, :, i])
                I = np.argsort(-np.real(w0))
                I = (np.hstack((I[0:self.m], I[channel_num - self.m:channel_num]))).tolist()
                self.csp_proj_matrix[:, :, i] = np.transpose(w1[:, I])
        return self

    def transform(self, data_x):
        """
        空间滤波后，得到信号的energy(variance)

        输入参数
        ----------
        data_x: T×N×L ndarray(或单个trial T×N)
               T: 采样点数  N: 通道数  L: 训练数据trial总数 B: 目标滤波段数
        csp_proj_matrix: 2m×N(2类csp)/2m×N×C(多类csp)
               投影矩阵
        返回值
        ----------
        x_after_csp: trial × feature
                    L×2m (或单个epoch 1D 2m)  多分类: L×(2m*C) (或单个epoch 1D (2m*C,))
                空间滤波后的数据取log作为特征
        """
        feature_len = self.csp_proj_matrix.shape[0]
        if data_x.ndim == 3:  # 多个trial
            trial_size = data_x.shape[2]
            if self.csp_proj_matrix.ndim == 3:  # 多分类
                class_num = self.csp_proj_matrix.shape[2]
                x_after_csp = np.zeros([trial_size, feature_len*class_num])
                for i in range(class_num):
                    for j in range(trial_size):
                        x_after_csp[j, (i*feature_len):((i+1)*feature_len)] = self._get_feature(self.csp_proj_matrix[:, :, i], data_x[:, :, j])
            else:  # 二分类
                x_after_csp = np.zeros([trial_size, feature_len])
                for i in range(trial_size):
                    x_after_csp[i, :] = self._get_feature(self.csp_proj_matrix, data_x[:, :, i])
        else:  # 单个trial
            if self.csp_proj_matrix.ndim == 3:  # 多分类
                class_num = self.csp_proj_matrix.shape[2]
                x_after_csp = np.zeros([1, feature_len*class_num])
                for i in range(class_num):
                    feature = self._get_feature(self.csp_proj_matrix[:, :, i], data_x)
                    x_after_csp[0, (i*feature_len):((i+1)*feature_len)] = feature
            else:  # 二分类
                feature = self._get_feature(self.csp_proj_matrix, data_x)
                x_after_csp = np.zeros([1, feature_len])
                x_after_csp[0, :] = feature
        return x_after_csp

    def fit_transform(self, train_x, train_y):
        return self.fit(train_x, train_y).transform(train_x)

    def _cov_matrix(self, train_x, train_y, class_list):
        # 计算协方差矩阵
        channel_num = train_x.shape[1]
        train_size = train_x.shape[2]
        class_num = len(class_list)
        R_one = np.zeros([channel_num, channel_num, class_num])  # 初始化
        R_rest = np.zeros([channel_num, channel_num, class_num])
        R_count = np.zeros(class_num)
        for i in range(train_size):
            x = train_x[:, :, i]
            R = np.dot(np.transpose(x), x) / np.trace(np.dot(np.transpose(x), x))
            for j in range(class_num):
                if train_y[i] == class_list[j]:
                    R_one[:, :, j] = R_one[:, :, j] + R  # 求和
                    R_count[j] = R_count[j] + 1
        for i in range(class_num):
            R_one[:, :, i] = R_one[:, :, i] / R_count[i]  # 协方差矩阵的归一化
        R_sum = np.sum(R_one, axis=2)
        for i in range(class_num):
            R_rest[:, :, i] = R_sum - R_one[:, :, i]
        return R_one, R_rest, R_sum

    def _get_feature(self, wk, eki):
        wk_t = np.transpose(wk)  # 转置
        eki_t = np.transpose(eki)
        z1 = np.dot(np.dot(np.dot(wk, eki_t), eki), wk_t)
        z2 = np.log(np.diag(z1) / np.trace(z1))
        return z2
