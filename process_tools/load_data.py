# -*- coding: utf-8 -*-
import random
import numpy as np


def loadnpz(filepath):
    # signal (samples, channels)
    # data_x(samples, channels, trials)
    file_data = np.load(filepath)
    signal = file_data.f.signal
    events = file_data.f.events
    fs = file_data.f.header_dict[()]['sample_rate']
    cue_dur = file_data.f.stim_pram_dict[()]['display_cue_duration']
    data_x = np.zeros([cue_dur * fs, signal.shape[1], events.shape[0]])
    data_y = np.zeros(events.shape[0], dtype=np.int)
    for i in range(events.shape[0]):
        data_x[:, :, i] = signal[events[i, 0]:events[i, 0] + cue_dur * fs, :]
        data_y[i] = events[i, 2]
    return data_x, data_y, fs

def loadnew_npz(filepath, config_path):
    # signal (samples, channels)
    # data_x(samples, channels, trials)
    file_data = np.load(filepath)
    signal = file_data.f.signal
    events = file_data.f.events
    fs = file_data.f.header_dict[()]['sample_rate']
    cue_dur = file_data.f.stim_pram_dict[()]['display_cue_duration']
    data_x = np.zeros([cue_dur * fs, signal.shape[1], events.shape[0]])
    data_y = np.zeros(events.shape[0], dtype=np.int)
    for i in range(events.shape[0]):
        data_x[:, :, i] = signal[events[i, 0]:events[i, 0] + cue_dur * fs, :]
        data_y[i] = events[i, 2]

    npz_data_dict = dict(np.load(filepath))
    data = npz_data_dict['signal']
    events = npz_data_dict['events']
    # self.stim_log = npz_data_dict['stim_log']
    config_data_dict = dict(np.load(config_path))
    event_id = config_data_dict['event_id_dict']
    fs = config_data_dict['nsheader_dict']['sample_rate']
    ch_names = config_data_dict['nsheader_dict']['channel_list']
    ch_types = config_data_dict['nsheader_dict']['channel_type']
    stim_pram = config_data_dict['stim_pram_dict']
    return data_x, data_y, fs


def slidingwin(data_x, data_y=None, step=500, window=500, is_shuffle=True):
    """
    滑窗  trial-->epoch , 乱序
    data_x: T×N×L ndarray 或单个trial T×N  T: 采样点数  N: 通道数  L: 训练数据 trial 总数 (n_times, n_ch, n_epochs)
    data_y: shape (n_samples,)L 个 trial 对应的标签
    data_x_epoch: T×N×L ndarray  T: 一个窗采样点数  N: 通道数  L: 训练数据 epoch 总数
    data_y_epoch: shape (n_samples,)L 个 epoch 对应的标签
    """
    delay = 250  # Start_Of_Trial后延迟0.5s
    samples, channal_num, trial_num = data_x.shape[0], data_x.shape[1], data_x.shape[2]
    window_num = int((samples - delay - window) / step + 1)  # 每个trial 滑多少个窗
    epoch_num = trial_num * window_num
    # data_y_epoch = np.zeros((epoch_num, 4))  # one-hot
    data_y_epoch = np.zeros(epoch_num)
    data_x_epoch = np.zeros([window, channal_num, epoch_num])
    epoch_count = 0
    for k in range(trial_num):
        for i in range(window_num):
            data_y_epoch[epoch_count] = data_y[k]
            start = int(i * step + delay)
            data_x_epoch[:, :, epoch_count] = data_x[start:start + window, :, k]
            epoch_count = epoch_count + 1
    # 打乱次序
    if is_shuffle:
        li = list(range(epoch_num))
        random.shuffle(li)
        data_x_epoch = data_x_epoch[:, :, li]
        data_y_epoch = data_y_epoch[li]
    return data_x_epoch, data_y_epoch
