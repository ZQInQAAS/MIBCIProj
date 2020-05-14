# -*- coding: utf-8 -*-
import numpy as np


def loadnpz(filepath):
    # signal (samples, channels)
    # data_x(samples, channels, trials)
    file_data = np.load(filepath, allow_pickle=True)
    signal = file_data.f.signal
    signal = np.delete(signal, [11, 12], 1)  # 移除f4 cp3
    events = file_data.f.events
    fs = file_data.f.header_dict[()]['sample_rate']
    cue_dur = file_data.f.stim_pram_dict[()]['display_cue_duration']
    data_x = np.zeros([cue_dur * fs, signal.shape[1], events.shape[0]])
    data_y = np.zeros(events.shape[0], dtype=np.int)
    for i in range(events.shape[0]):
        data_x[:, :, i] = signal[events[i, 0]:events[i, 0] + cue_dur * fs, :]
        data_y[i] = events[i, 2]
    return data_x, data_y, fs

def data2mne_raw(filepath):
    file_data = np.load(filepath, allow_pickle=True)
    signal = file_data.f.signal
    events = file_data.f.events
    event_id = file_data.f.event_id_dict[()]
    header_dict = file_data.f.header_dict[()]
    stim_pram_dict = file_data.f.stim_pram_dict[()]
    stim_log = file_data.f.stim_log
    fs = header_dict['sample_rate']
    pass