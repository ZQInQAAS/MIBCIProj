import mne
from philistine.mne import savgol_iaf, attenuation_iaf
import numpy as np
from scipy.integrate import simps
from process_tools import LazyProperty
from BCIConfig import ch_names, ch_types, event_id, fs
from mne.time_frequency import psd_welch, psd_multitaper


def cal_power_feature(signal, ch, freq_min=8, freq_max=30, rp=False):
    # signal (n_times, n_channels)  计算某频带的信号在指定通道的均值
    info = mne.create_info(ch_names, fs, ch_types)
    info.set_montage('standard_1005')
    raw_mne = mne.io.RawArray(signal.T, info, verbose=0)  # RawArray input (n_channels, n_times)
    raw_mne.set_eeg_reference(ref_channels='average', projection=True, verbose=0).apply_proj()  # CAR
    # raw_mne.filter(1, 100, verbose=0)  # band pass
    raw_mne = raw_mne.pick_channels(list(ch))
    raw_mne = raw_mne.reorder_channels(list(ch))
    # psd, freqs = psd_welch(raw_mne, fmin=fmin, fmax=fmax, proj=True, verbose='warning')  # psd (channal, freq)
    # psd_multitaper normalization='full' PSD将由采样率和信号长度归一化
    psd, freqs = psd_multitaper(raw_mne, fmin=freq_min, fmax=freq_max, adaptive=True, normalization='full', verbose=0)
    freq_res = freqs[1] - freqs[0]  # 频率分辨率
    power_ch = simps(psd, dx=freq_res)  # 求积分 power:(channal,)
    power = np.average(power_ch)  # 通道平均
    if rp:
        psd, freqs = psd_multitaper(raw_mne, fmin=2, fmax=50, adaptive=True, normalization='full', verbose=0)
        freq_res = freqs[1] - freqs[0]  # 频率分辨率
        power_ch = simps(psd, dx=freq_res)  # 求积分 power:(channal,)
        power_all = np.average(power_ch)  # 通道平均
        power_rp = power / power_all
        return power, power_rp
    else:
        return power


class MIdataset(object):
    def __init__(self, path):
        self.read_newdata(path)
        self.epoch_data = None
        self._info = None
        self._raw_mne = None
        self._epochs_mne = None
        self.mne_scallings = dict(eeg=20, eog=500)

    def read_newdata(self, path):
        npz_data_dict = dict(np.load(path, allow_pickle=True))
        self.data = npz_data_dict['signal']
        self.events = npz_data_dict['events']
        # self.stim_log = npz_data_dict['stim_log']
        self.ch_names =ch_names
        self.ch_types = ch_types
        self.event_id = event_id

    @LazyProperty
    def info(self):
        montage = 'standard_1020'
        self._info = mne.create_info(self.ch_names, fs, self.ch_types)
        self._info.set_montage(montage)
        return self._info

    @LazyProperty
    def raw_mne(self):
        self.info['events'] = self.events
        self._raw_mne = mne.io.RawArray(self.data.T, self.info)  # RawArray input (n_channels, n_times)
        # self._raw_mne.pick_types(eog=False, eeg=True)
        return self._raw_mne

    @LazyProperty
    def epochs_mne(self):
        # epochs: (n_epochs, n_chans, n_times)
        # reject_criteria = dict(eeg=150e-6, eog=250e-6)  # eeg150 µV  eog250 µV Exclude the signal with large amplitude
        self._epochs_mne = mne.Epochs(self.raw_mne, self.events, self.event_id,
                                      tmin=-4, tmax=5, baseline=None, preload=True)
        return self._epochs_mne

    def get_raw_data(self):
        # return (samples, channels)
        if self._raw_mne:
            return self.raw_mne.get_data(picks='eeg').T  # get_data (n_channels, n_times)
        else:
            return self.data[:, :-2]

    def get_epoch_data(self, tmin=0, tmax=5, select_label=None, select_ch=None):
        # return: (sample, channel, trial) label:left right foot rest
        epochs = self.epochs_mne.crop(tmin=tmin, tmax=tmax)
        if select_ch:
            epochs = epochs.pick_channels(select_ch)
            epochs = epochs.reorder_channels(select_ch)
        epochs.pick_types(eeg=True)
        epochs = epochs[select_label]
        label = epochs.events[:, -1]
        data = epochs.get_data()  # return (n_epochs, n_channels, n_times)
        data = data.swapaxes(0, 2)
        return data, label

    def get_IAF(self, raw=None, ch=None, fmin=None, fmax=None, res=0.25):
        if raw is None:
            raw = self.raw_mne
        iaf = savgol_iaf(raw, picks=ch, fmin=fmin, fmax=fmax, resolution=res, ax=False)
        return iaf.PeakAlphaFrequency

    def plot_raw_psd(self, fmin=1, fmax=40):
        self.raw_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        self.raw_mne.plot(duration=5, n_channels=28, show=True, scalings=self.mne_scallings)
        self.raw_mne.plot_psd(fmin, fmax, n_fft=2 ** 10, spatial_colors=True)

    def plot_events(self):
        mne.viz.plot_events(self.events, event_id=self.event_id, sfreq=fs, first_samp=self.raw_mne.first_samp)

    def plot_epoch(self, fmin=1, fmax=40):
        self.epochs_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        self.epochs_mne.plot(scalings=self.mne_scallings, n_epochs=5, n_channels=28)
        self.epochs_mne.plot_psd(fmin, fmax, average=True, spatial_colors=True)
        self.epochs_mne.plot_psd_topomap()

    def plot_oneclass_epoch(self, event_name, ch_name):
        oneclass_epoch = self.epochs_mne[event_name]  # left/right/foot/rest
        oneclass_epoch.plot_image(picks=ch_name)
        self.epochs_mne.plot_psd_topomap()

    def plot_tf_analysis(self, event_name, ch_name):
        # time-frequency analysis via Morlet wavelets
        event_epochs = self.epochs_mne[event_name]
        frequencies = np.arange(7, 30, 3)
        power = mne.time_frequency.tfr_morlet(event_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3)
        power.plot(picks=ch_name, baseline=(-2, 0), mode='logratio', title=event_name + ch_name)

    def bandpass_filter(self, l_freq=1, h_freq=40):
        self.raw_mne.filter(l_freq, h_freq, verbose='warning')

    def set_reference(self, ref='average'):
        self.raw_mne.set_eeg_reference(ref_channels=ref, projection=True).apply_proj()  #CAR

    def removeEOGbyICA(self):
        # 根据EOG寻找相似IC 去眼电
        ica = mne.preprocessing.ICA(n_components=12, random_state=97, max_iter=800)
        ica.fit(self.raw_mne)
        ica.exclude = []
        eog_indices, eog_scores = ica.find_bads_eog(self.raw_mne, threshold=2.3)
        ica.exclude = eog_indices
        # orig_raw = self.raw_mne.copy()
        ica.apply(self.raw_mne)


if __name__ == '__main__':
    # path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean\S4\S4_20200721' \
    #        r'\NSsignal_2020_07_21_16_05_04.npz'
    path = r'D:\Myfiles\MIBCI_NF\data_set\wmm\wmm_20210126\Online_20210126_1525_33.npz'
    # config_p =r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\config.npz'
    d = MIdataset(path)
    PAF, AB = d.get_IAF()
    d.plot_tf_analysis('left', 'C3')