import mne
import numpy as np
import pandas as pd
import pickle
from scipy.integrate import simps
from mne.connectivity import spectral_connectivity
from philistine.mne import savgol_iaf, attenuation_iaf
from BCIConfig import ch_names, ch_types, events_id_3mi, fs
from mne.time_frequency import psd_welch, psd_multitaper


class MIdataset(object):
    def __init__(self):
        self.mne_scallings = dict(eeg=20, eog=500)

    def read_rawdata_path(self, path):
        npz_data_dict = dict(np.load(path, allow_pickle=True))  # npz
        data = npz_data_dict['signal']  # (n_times, n_ch)
        events = npz_data_dict['events']
        # self.epoch_data = npz_data_dict['epochs']
        # stim_log = npz_data_dict['stim_log']
        return data, events

    def read_cleandata_path(self, path):
        with open(path, 'rb') as f:
            epochs_mne = pickle.load(f)  # pkl
        return epochs_mne

    def gather_data(self, npzpath_list):
        data_gather, events_gather = self.read_rawdata_path(npzpath_list[0])
        for i in range(1, len(npzpath_list)):
            data, events = self.read_rawdata_path(npzpath_list[i])
            events[:, 0] = events[:, 0] + data_gather.shape[0]
            data_gather = np.concatenate((data_gather, data), axis=0)
            events_gather = np.concatenate((events_gather, events), axis=0)
        return data_gather, events_gather

    def gather_epoch(self, pklpath_list):
        epoch_list = []
        for i in range(len(pklpath_list)):
            epochs_mne = self.read_cleandata_path(pklpath_list[i])
            epoch_list.append(epochs_mne)
        epochs_mne = mne.concatenate_epochs(epoch_list)
        return epochs_mne

    def get_dictdata(self, path_list):
         # fbcsp
        data, events = self.gather_data(path_list)
        raw_mne = self.get_raw_mne(data.T)
        raw_mne = self.drop_channel(raw_mne, ch=['M1', 'M2'])
        # raw_mne = self.set_reference(raw_mne)  # CAR
        epochs_mne = self.get_epochs_mne(raw_mne, events=events, event_id=events_id_3mi, tmin=-4, tmax=4)
        epochs_mne = self.removeEOGbyICA(epochs_mne)  # ICA
        epochs_mne = self.drop_channel(epochs_mne, ch=['HEOG', 'VEOG'])
        x_data = epochs_mne.get_data()
        y_labels = events[:, -1] - min(events[:, -1])
        return {'x_data': x_data, 'y_labels': y_labels, 'fs': fs}

    def get_info(self, fs=fs, ch_names=ch_names, ch_types=ch_types, montage='standard_1020'):
        info = mne.create_info(ch_names, fs, ch_types, verbose=0)
        info.set_montage(montage)
        return info

    def get_raw_mne(self, raw_array, ch_names=ch_names, ch_types=ch_types):
        # raw_array: RawArray input (n_ch, n_times)
        info = self.get_info(ch_names=ch_names, ch_types=ch_types)
        # info['events'] = events
        raw_mne = mne.io.RawArray(raw_array, info, verbose=0)
        # self._raw_mne.pick_types(eog=False, eeg=True)
        return raw_mne

    def get_epochs_mne(self, raw_mne, events=None, event_id=None, tmin=-5, tmax=4):
        # epochs_mne.get_data: (n_epochs, n_chans, n_times)
        # reject_criteria = dict(eeg=150e-6, eog=250e-6)  # eeg150 µV  eog250 µV Exclude the signal with large amplitude
        epochs_mne = mne.Epochs(raw_mne, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                                     baseline=None, preload=True, verbose=0)
        return epochs_mne

    def get_epochs_mne_byarray(self, epoch_array, ch_name=ch_names, ch_type=ch_types, events=None, event_id=None, tmin=0):
        # epoch_array: (n_epochs, n_chans, n_times)
        info = self.get_info(ch_names=ch_name, ch_types=ch_type,)
        # reject_criteria = dict(eeg=150e-6, eog=250e-6)  # eeg150 µV  eog250 µV Exclude the signal with large amplitude
        epochs_mne = mne.EpochsArray(epoch_array, info, events, event_id=event_id, tmin=tmin,
                                     baseline=None, verbose=0)
        return epochs_mne

    def get_raw_data(self, raw_mne):
        # return (samples, channels)
        return raw_mne.get_data(picks='eeg').T  # get_data (n_channels, n_times)

    def get_epoch_array(self, epochs_mne, tmin=-5, tmax=4, select_label=None, select_ch=None):
        epochs = epochs_mne.crop(tmin=tmin, tmax=tmax)
        if select_ch:
            epochs = self.pick_channel(epochs, select_ch)
        epochs.pick_types(eeg=True)
        if select_label:
            epochs = epochs[select_label]  # label:left right foot rest
        label = epochs.events[:, -1]
        epoch_array = epochs.get_data()  # epoch_array (n_epochs, n_ch, n_times)
        # epoch_array = epoch_array.swapaxes(0, 2) # 交换一三维  (n_times, n_ch, n_epochs)
        return epoch_array, label

    def get_IAF(self, raw_mne, ch=None, fmin=None, fmax=None, res=0.25, ax=False):
        iaf = savgol_iaf(raw_mne, picks=ch, fmin=fmin, fmax=fmax, resolution=res, ax=ax)
        return iaf.PeakAlphaFrequency, iaf.AlphaBand

    def get_power(self, inst_mne, fmin, fmax, tmin=None, tmax=None, label=None, pick_ch=None):
        if label:
            inst_mne = inst_mne[label]
        if pick_ch:
            inst_mne = self.pick_channel(inst_mne, pick_ch)
        # 输入inst_mne为raw/单个epoch时psd为(channal, freq; 多个epoch时psd为(trial, channal, freq)
        # psd, freqs = psd_welch(raw_mne, fmin=fmin, fmax=fmax, proj=True, verbose='warning')
        psd, freqs = psd_multitaper(inst_mne, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, adaptive=True,
                                    normalization='full', verbose=0)  # normalization='full'通过采样率和信号长度将PSD归一化
        freq_res = freqs[1] - freqs[0]  # 频率分辨率
        power_ch = simps(psd, dx=freq_res)  # 求积分 power_ch:(channal,) or (trial, channal)
        if power_ch.ndim == 2:
            power = np.average(power_ch, axis=0)  # 按trial求和 (channal,)
        else:
            # power = np.average(power_ch)  # 按通道平均 1D
            power = power_ch
        return power

    def get_rela_power(self, inst_mne, fmin, fmax, basefmin=4, basefmax=50, tmin=None, tmax=None, label=None, pick_ch=None):
        power = self.get_power(inst_mne, fmin, fmax, tmin, tmax, label, pick_ch)
        power_all = self.get_power(inst_mne, basefmin, basefmax, tmin, tmax, label, pick_ch)
        rela_power = power / power_all
        return power, rela_power

    def get_power_byarray(self, signal, ch, fmin=8, fmax=30):
        # signal (n_ch, n_times)  计算某频带的信号在指定通道的均值
        raw_mne = self.get_raw_mne(signal)
        power = self.get_power(raw_mne, fmin, fmax, pick_ch=ch)
        return power

    def get_relapower_byarray(self, signal, ch=None, fmin=8, fmax=30, basefmin=1, basefmax=50):
        # signal (n_ch, n_times)，power: float
        raw_mne = self.get_raw_mne(signal)
        power = self.get_power(raw_mne, fmin, fmax, pick_ch=ch)
        power_all = self.get_power(raw_mne, fmin=basefmin, fmax=basefmax, pick_ch=ch)
        power_rp = power / power_all
        return power, power_rp

    def get_spectral_connectivity(self, epochs_mne, fmin=0, fmax=np.inf, tmin=0, method='pli'):
        # 功能性连接  con (n_signals, n_signals)
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs_mne, method=method, mode='multitaper', sfreq=fs, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1, verbose='warning')
        con = con[:, :, 0]
        return con

    def pick_channel(self, inst_mne, ch):
        inst_mne = inst_mne.pick_channels(list(ch))
        inst_mne = inst_mne.reorder_channels(list(ch))
        return inst_mne

    def drop_channel(self, inst_mne, ch):
        inst_mne = inst_mne.drop_channels(list(ch))
        return inst_mne

    def bandpass_filter(self, raw_mne, l_freq=1, h_freq=40):
        raw_mne.filter(l_freq, h_freq, verbose=0)
        return raw_mne

    def set_reference(self, raw_mne, ref='average'):
        raw_mne.set_eeg_reference(ref_channels=ref, projection=True, verbose=0).apply_proj()  # CAR
        return raw_mne

    def removeEOGbyICA(self, inst_mne):
        # 根据EOG寻找相似IC 去眼电
        ica = mne.preprocessing.ICA(n_components=12, random_state=97, max_iter=800, verbose=0)
        ica.fit(inst_mne, verbose=0)
        ica.exclude = []
        eog_indices, eog_scores = ica.find_bads_eog(inst_mne, threshold=2.3, verbose=0)
        ica.exclude = eog_indices
        # orig_raw = self.raw_mne.copy()
        ica.apply(inst_mne)
        return inst_mne

    def plot_raw_psd(self, raw_mne, fmin=0, fmax=120):
        raw_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        raw_mne.plot(duration=5, n_channels=28, show=True, scalings=self.mne_scallings)
        raw_mne.plot_psd(fmin, fmax, n_fft=2 ** 10, spatial_colors=True)

    def plot_events(self, events, first_samp):
        # first_samp = self.raw_mne.first_samp
        mne.viz.plot_events(events, event_id=events_id_3mi, sfreq=fs, first_samp=first_samp)

    def plot_epoch(self, epochs_mne, fmin=1, fmax=40):
        epochs_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        epochs_mne.plot(scalings=self.mne_scallings, n_epochs=5, n_channels=28)
        epochs_mne.plot_psd(fmin, fmax, average=True, spatial_colors=True)
        epochs_mne.plot_psd_topomap()

    def plot_oneclass_epoch(self, epochs_mne, event_name, ch_name):
        oneclass_epoch = epochs_mne[event_name]  # left/right/foot/rest
        oneclass_epoch.plot_image(picks=ch_name)
        epochs_mne.plot_psd_topomap()

    def plot_tf_analysis(self, epochs_mne, event_name, ch_name):
        # time-frequency analysis via Morlet wavelets
        event_epochs = epochs_mne[event_name]
        frequencies = np.arange(7, 30, 3)
        power = mne.time_frequency.tfr_morlet(event_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3)
        power.plot(picks=ch_name, baseline=(-2, 0), mode='logratio', title=event_name + ch_name)


if __name__ == '__main__':
    # path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean\S4\S4_20200721' \
    #        r'\NSsignal_2020_07_21_16_05_04.npz'
    path = r'/data_set_clean/PNM2\\PNM_20210607\\Acq_post_20210607_1130_31.pkl'
    # path = [r'D:\Myfiles\MIBCI_NF\data_set\LXT\LXT_20210622\Acq_post_20210622_1455_24.npz',
    #         r'D:\Myfiles\MIBCI_NF\data_set\LXT\LXT_20210622\Acq_post_20210622_1502_16.npz',
    #         r'D:\Myfiles\MIBCI_NF\data_set\LXT\LXT_20210622\Acq_post_20210622_1508_54.npz']
    # config_p =r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\config.npz'
    MI = MIdataset()
    # a = MI.get_dictdata(path)
    epoch = MI.read_cleandata_path(path)
    data, events = MI.read_rawdata_path(path)
    # raw_mne = MI.get_raw_mne(data.T)
    # raw_mne = MI.removeEOGbyICA(raw_mne)  # ICA
    # d.bandpass_filter(2, 100)
    # MI.plot_raw_psd(raw_mne)
    # PAF, AB = d.get_IAF()
    # d.plot_tf_analysis('left', 'C3')
