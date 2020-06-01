import mne
import numpy as np
from utils import LazyProperty


class data_4class(object):
    def __init__(self, path):
        npz_data = np.load(path)
        self.fs = npz_data.f.SampleRate[0]
        self.index = npz_data.f.MarkOnSignal
        self.index[:, 0] = self.index[:, 0] - 6000
        data = npz_data.f.signal  # (samples, channels)
        self.data = data[5999:, 0:-1]  # 去事件通道
        self.events = None
        self.epoch_data = None
        self._raw_mne = None
        self._epochs_mne = None
        self.mne_scallings = dict(eeg=20, eog=500)

    @LazyProperty
    def raw_mne(self):
        info = self.create_info()
        self._raw_mne = mne.io.RawArray(self.data.T, info)  # RawArray input (n_channels, n_times)
        return self._raw_mne

    @LazyProperty
    def epochs_mne(self):
        # epochs: (n_epochs, n_chans, n_times)
        if self.events is None:
            self._get_epoch_event()
        # reject_criteria = dict(eeg=150e-6, eog=250e-6)  # eeg150 µV  eog250 µV 排除振幅过大信号
        self._epochs_mne = mne.Epochs(self.raw_mne, self.events, self.event_id, tmin=-2, tmax=5, baseline=None,  preload=True)
        return self._epochs_mne

    def psd_multitaper(self, type=None, fmin=0, fmax=np.inf, tmin=None, tmax=None):
        type_dict=dict(raw=self.raw_mne, epoch=self.epochs_mne)
        inst = type_dict[type] if type else self.raw_mne
        psds, freqs = mne.time_frequency.psd_multitaper(inst, low_bias=True, tmin=tmin, tmax=tmax,
                                     fmin=fmin, fmax=fmax, proj=True, picks='eeg', n_jobs=1)
        return psds, freqs

    def _get_epoch_event(self, tmin=0, tmax=5):
        # epoch_data: n_samples, n_channal, n_trial
        j = 0
        delete_epoch_idx = []
        channal_num = self.data.shape[1]
        trial_num = len(np.where(self.index[:, 1] == 800)[0])
        self.epoch_data = np.zeros([(tmax-tmin)*self.fs, channal_num, trial_num])
        self.events = np.zeros([trial_num, 3], dtype=np.int)
        for i in range(self.index.shape[0]):
            if self.index[i, 1] in [769, 770, 771, 780]:
                try:
                    start = self.index[i, 0] + tmin * self.fs
                    end = self.index[i, 0] + tmax * self.fs
                    self.epoch_data[:, :, j] = self.data[start:end, :]
                    self.events[j, 0] = self.index[i, 0]
                    self.events[j, 2] = self.index[i, 1] - 768
                    j += 1
                except ValueError:
                    delete_epoch_idx.append(j)  # 丢包
                except IndexError:
                    print('Index ', j, ' out of bounds, incomplete data file!')  # 实验中断
                    break
        if delete_epoch_idx:
            print('data lost at trial', delete_epoch_idx)
            self.epoch_data = np.delete(self.epoch_data, delete_epoch_idx, 2)
            self.events = np.delete(self.events, delete_epoch_idx)
        self.events[np.where(self.events[:, 2] == 12), 2] = 4
        self.event_id = dict(left=1, right=2, foot=3, rest=4)
        # label = events[:, -1]

    def get_raw_data(self):
        # return (samples, channels)
        if self._raw_mne:
            return self.raw_mne.get_data(picks='eeg').T  # get_data (n_channels, n_times)
        else:
            return self.data[:, :-2]

    def set_raw_data(self, raw_data):
        # (samples, channels)
        self.eog = self.data[:, -2:]
        self.data = np.concatenate((raw_data, self.eog), axis=1)

    def get_epoch_data(self, tmin=0, tmax=5):
        # return: (sample, channel, trial)
        if self._epochs_mne:
            label = self.epochs_mne.events[:, -1]
            epochs = self.epochs_mne.crop(tmin=tmin, tmax=tmax)
            data = epochs.get_data(picks='eeg')  # (n_epochs, n_channels, n_times)
            data = data.swapaxes(0, 2)
        else:
            if self.epoch_data is None:
                self._get_epoch_event(tmin, tmax)
            data = self.epoch_data
            label = self.events[:, -1]
        # data = np.delete(data, [4, 20], 1)  # 移除f4 cp3
        return data, label

    def set_epoch_data(self, epochs_data):
        # (sample, channel, trial)
        self.epochs_data = epochs_data

    def create_info(self):
        channel_names = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3',
                         'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'hEOG', 'vEOG']
        n_channels = len(channel_names) - 2
        channel_types = ['eeg'] * n_channels + ['eog'] * 2
        montage = 'standard_1005'
        info = mne.create_info(channel_names, self.fs, channel_types)
        info.set_montage(montage)
        return info

    def plot_raw_psd(self, fmin=1, fmax=40):
        self.raw_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        self.raw_mne.plot(duration=5, n_channels=28, show=True, scalings=self.mne_scallings)
        self.raw_mne.plot_psd(fmin, fmax, n_fft=2 ** 10, spatial_colors=True)  # 功率谱

    def plot_events(self):
        if self.events is None:
            self._get_epoch_event()
        mne.viz.plot_events(self.events, event_id=self.event_id, sfreq=self.fs, first_samp=self.raw_mne.first_samp)

    def plot_epoch(self, fmin=1, fmax=40):
        self.epochs_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        self.epochs_mne.plot(scalings=self.mne_scallings, n_epochs=5, n_channels=28)
        self.epochs_mne.plot_psd(fmin, fmax, average=True, spatial_colors=True)
        self.epochs_mne.plot_psd_topomap()

    def plot_oneclass_epoch(self, event_name, ch_name):
        oneclass_epoch = self.epochs_mne[event_name]
        oneclass_epoch.plot_image(picks=ch_name)
        self.epochs_mne.plot_psd_topomap()

    def plot_tf_analysis(self, event_name, ch_name):
        # time-frequency analysis via Morlet wavelets
        event_epochs = self.epochs_mne[event_name]
        frequencies = np.arange(7, 30, 3)
        power = mne.time_frequency.tfr_morlet(event_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3)
        power.plot(picks=ch_name, baseline=(-2, 0), mode='logratio', title=event_name+ch_name)

    def bandpass_filter(self, l_freq=1, h_freq=40):
        self.raw_mne.filter(l_freq, h_freq, verbose='warning')
        # filter_params = mne.filter.create_filter(self.raw_mne.get_data(), self.fs, l_freq=l_freq, h_freq=h_freq)
        # mne.viz.plot_filter(filter_params, self.fs, flim=(0.01, 5))  # plot 滤波器

    def notch_filter(self):
        self.raw_mne.notch_filter(freqs=np.arange(50, 250, 50), notch_widths=1)  # ?

    def set_reference(self, ref='average'):
        # ref: CAR 或 ['A1']
        self.raw_mne.set_eeg_reference(ref_channels=ref, projection=True).apply_proj()

    def set_CSD(self):
        raw_csd = mne.preprocessing.compute_current_source_density(self.raw_mne)
        self.raw_mne.plot(duration=5, n_channels=28, show=True, scalings=self.mne_scallings)
        raw_csd.plot(duration=5, n_channels=28, show=True, scalings=dict(csd=3500, eog=500))
        self.raw_mne.plot_psd(1, 40, n_fft=2 ** 10, spatial_colors=True)
        raw_csd.plot_psd(1, 40, n_fft=2 ** 10, spatial_colors=True)
        return raw_csd

    def plot_ICA_manually(self):
        # 手动选择伪迹IC 并去除
        ica = mne.preprocessing.ICA(n_components=12, random_state=97, max_iter=800)
        ica.fit(self.raw_mne)
        ica.plot_sources(inst=self.raw_mne)
        ica.plot_components(inst=self.raw_mne)
        orig_raw = self.raw_mne.copy()
        ica.apply(self.raw_mne)
        orig_raw.plot(start=0, duration=5, n_channels=28, block=False, scalings=self.mne_scallings)
        self.raw_mne.plot(start=0, duration=5, n_channels=28, block=True, scalings=self.mne_scallings)

    def removeEOGbyICA(self):
        # 根据EOG寻找相似IC 去眼电
        ica = mne.preprocessing.ICA(n_components=12, random_state=97, max_iter=800)
        ica.fit(self.raw_mne)
        ica.exclude = []
        eog_indices, eog_scores = ica.find_bads_eog(self.raw_mne, threshold=2.3)
        ica.exclude = eog_indices
        # orig_raw = self.raw_mne.copy()
        ica.apply(self.raw_mne)
        # ica.plot_scores(eog_scores)
        # ica.plot_properties(orig_raw, picks=eog_indices)  # plot diagnostics
        # ica.plot_sources(orig_raw)  # plot ICs applied to raw data, with EOG matches highlighted
        # eog_epoch = mne.preprocessing.create_eog_epochs(orig_raw, tmin=-1.5, tmax=1.5, thresh=200).average()
        # eog_epoch.apply_baseline(baseline=(-1, -0.5))
        # eog_epoch.plot_joint()
        # ica.plot_sources(eog_epoch)
        # orig_raw.plot(start=0, duration=5, n_channels=28, block=False, scalings=self.mne_scallings)
        # self.raw_mne.plot(start=0, duration=5, n_channels=28, block=True, scalings=self.mne_scallings)
