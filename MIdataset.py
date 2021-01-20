import mne
import numpy as np
from process_tools import LazyProperty

ch_names = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
            'CP4', 'CP6', 'hEOG', 'vEOG', 'M1', 'M2', 'F5', 'AF3', 'AF4', 'P5', 'P3', 'P1',
            'Pz', 'P2', 'P4', 'P6', 'PO3', 'POz', 'PO4', 'Oz', 'F6']  # 41ch EEG + 2ch EOG + 2ch reference
ch_types = ['eeg'] * 26 + ['eog'] * 2 + ['eeg'] * 2 + ['eeg'] * 15
# ch_types = ['eeg'] * 26 + ['eeg'] * 15


class MIdataset(object):
    def __init__(self, path):
        self.path = path
        npz_data = np.load(self.path)
        self.npz_data_dict = dict(npz_data)
        self.read_data()
        self.events = None
        self.epoch_data = None
        self._info = None
        self._raw_mne = None
        self._epochs_mne = None
        self.mne_scallings = dict(eeg=20, eog=500)

    def read_data(self):
        self.fs = self.npz_data_dict['SampleRate'][0]
        self.index = self.npz_data_dict['MarkOnSignal']
        data = self.npz_data_dict['signal']  # (samples, channels)
        self.data = data[:, 0:28] if data.shape[1] == 29 \
            else np.delete(data, [45], 1)  # 26, 27, 28, 29,EOG remove 28(M1) 29(M2) 45(event ch)
        self.ch_names, self.ch_types = (ch_names[:28], ch_types[:28]) \
            if self.data.shape[1] == 28 else (ch_names, ch_types)

    @LazyProperty
    def info(self):
        self._info = self.create_info()
        return self._info

    @LazyProperty
    def raw_mne(self):
        self._get_epoch_event()
        self.info['events'] = self.events
        self._raw_mne = mne.io.RawArray(self.data.T, self.info)  # RawArray input (n_channels, n_times)
        # self._raw_mne.pick_types(eog=False, eeg=True)
        self._raw_mne = self._raw_mne.drop_channels(['M1', 'M2'])
        return self._raw_mne

    @LazyProperty
    def epochs_mne(self):
        # epochs: (n_epochs, n_chans, n_times)
        if self.events is None:
            self._get_epoch_event()
        # reject_criteria = dict(eeg=150e-6, eog=250e-6)  # eeg150 µV  eog250 µV Exclude the signal with large amplitude
        self._epochs_mne = mne.Epochs(self.raw_mne, self.events, self.event_id, tmin=-4, tmax=5, baseline=None,
                                      preload=True)
        return self._epochs_mne

    def _get_epoch_event(self, tmin=-4, tmax=5):
        # epoch_data: n_samples, n_channal, n_trial
        j = 0
        delete_epoch_idx = []
        channal_num = self.data.shape[1]
        trial_num = len(np.where(self.index[:, 1] == 800)[0])
        self.epoch_data = np.zeros([(tmax - tmin) * self.fs, channal_num, trial_num])
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
                    delete_epoch_idx.append(j)  # Data Loss
                except IndexError:
                    print(self.path)
                    print('Index ', j, ' out of bounds, incomplete data file!')  # The experimental interrupt
                    break
        if delete_epoch_idx:
            print(self.path)
            print('data lost at trial', delete_epoch_idx)
            self.epoch_data = np.delete(self.epoch_data, delete_epoch_idx, 2)
            self.events = np.delete(self.events, delete_epoch_idx, 0)
        self.events[np.where(self.events[:, 2] == 12), 2] = 4
        self.event_id = dict(left=1, right=2, foot=3, rest=4)
        # label = events[:, -1]

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

    def create_info(self):
        montage = 'standard_1005'
        info = mne.create_info(self.ch_names, self.fs, self.ch_types)
        info.set_montage(montage)
        return info

    def plot_raw_psd(self, fmin=1, fmax=40):
        self.raw_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        self.raw_mne.plot(duration=5, n_channels=28, show=True, scalings=self.mne_scallings)
        self.raw_mne.plot_psd(fmin, fmax, n_fft=2 ** 10, spatial_colors=True)

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
    path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean\S4\S4_20200721\NSsignal_2020_07_21_16_05_04.npz'
    path = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\S1\S1_20210119\acq_20210119_2046_51.npz'
    d = MIdataset(path)
    data, label = d.get_epoch_data(select_label=['left', 'right'], select_ch=['F3', 'F1', 'F2', 'F4', 'FC3', 'FC5'])
    # d.set_reference()
    # d.plot_raw_psd()
    # d.plot_events()
    # d.plot_epoch()
    d.plot_tf_analysis('left', 'C3')