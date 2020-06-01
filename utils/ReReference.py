# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np


def CARFilter(data_x):
    """
    共均值参考滤波common average referencing filter
    data_x: T×N×L ndarray(或单个trial T×N)
           T: 采样点数  N: 通道数  L: 训练数据trial总数
    AfterCAR_data_x: CAR后的数据
    """
    channel_num = data_x.shape[1]
    after_car_data_x = np.zeros(data_x.shape)
    if data_x.ndim == 3:
        trial_size = data_x.shape[2]
        for j in range(trial_size):
            channel_sum = np.zeros(data_x.shape[0])
            for i in range(channel_num):
                channel_sum = channel_sum + data_x[:, i, j]
            channel_mean = channel_sum / channel_num
            for i in range(channel_num):
                after_car_data_x[:, i, j] = data_x[:, i, j] - channel_mean
    else:
        channel_sum = np.zeros(data_x.shape[0])
        for i in range(channel_num):
            channel_sum = channel_sum + data_x[:, i]
        channel_mean = channel_sum / channel_num
        for i in range(channel_num):
            after_car_data_x[:, i] = data_x[:, i] - channel_mean
    return after_car_data_x
