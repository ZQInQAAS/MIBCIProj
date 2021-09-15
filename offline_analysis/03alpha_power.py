import os
import numpy as np
from BCIConfig import pick_rest_ch, fs, ch_names60, ch_types60
from MIdataset_NF import MIdataset

def load_model(path):
    baseline_model = dict(np.load(path, allow_pickle=True))
    # alpha_band = baseline_model['alpha_band']
    PAF = baseline_model['PAF']
    alpha_band = (PAF-2, PAF+2)
    # alpha_band = (7, 14)
    # delta = (1, 3.5), theta =(IAF-6, IAF-2),
    # alpha = (IAF-2, IAF+2),
    # beta1 = (IAF+2, IAF+8), beta2 = (IAF+8,IAF+14), beta3 = (IAF+14, 30).
    # beta1 = (IAF+2, IAF+11), beta2 = (IAF+11,IAF+20), low gamma = (30, 50)
    print('PAF', PAF, 'band:', alpha_band)
    return alpha_band

def cal_power(path, alpha_band):
    MI = MIdataset()
    eo_baseline_pre = dict(np.load(path, allow_pickle=True))
    baseline_eo = eo_baseline_pre['signal']
    fmin, fmax = alpha_band[0], alpha_band[1]
    # data1 = baseline_eo[2500:27505, :]  # 50s
    power0, rela_power0 = MI.get_relapower_byarray(baseline_eo.T, pick_rest_ch, fmin=fmin, fmax=fmax, basefmin=1)
    power, rela_power = np.average(power0), np.average(rela_power0)
    # epochs_mne = load_epoch(baseline_eo)
    # power_ch, rela_power_ch = MI.get_rela_power(epochs_mne, fmin, fmax, basefmin=1, basefmax=50,
    #                                       tmin=0, tmax=5, pick_ch=pick_rest_ch, label='1')
    # power, rela_power = np.average(power_ch), np.average(rela_power_ch)
    print('power:', round(power, 3), 'Baseline_power:', round(rela_power, 3))
    # print('power:', round(power, 3))
    return power, rela_power

def load_epoch(data):
    MI = MIdataset()
    raw_mne = MI.get_raw_mne(data.T)
    raw_mne = MI.drop_channel(raw_mne, ch=['M1', 'M2'])
    raw_mne = MI.drop_channel(raw_mne, ch=['HEOG', 'VEOG'])
    tmin, tmax = 0, 5
    n_epochs, st = 10, 5  # st开始时间
    start, stop = st * fs, (st + n_epochs * tmax) * fs + n_epochs
    raw_array = raw_mne.get_data(start=int(start), stop=int(stop))  # (n_ch, n_times)
    epoch_array = raw_array.reshape(60, n_epochs, -1)  # (n_ch, n_epochs, n_times)
    epoch_array = epoch_array.swapaxes(0, 1)  # (n_epochs, n_ch, n_times)
    epochs_mne = MI.get_epochs_mne_byarray(epoch_array, ch_name=ch_names60, ch_type=ch_types60, tmin=tmin)
    return epochs_mne

if __name__ == '__main__':
    p = os.path.abspath(os.path.dirname(os.getcwd()))
    p = p + r'\data_set\CYJ\CYJ_20210703'
    path1 = p + r'\Baseline_pre_eo.npz'
    # path2 = p + r'\Baseline_pre_ec.npz'
    path3 = p + r'\Baseline_post_eo.npz'
    # path4 = p + r'\Baseline_post_ec.npz'
    model_path = p + r'\model.npz'
    IAF_band = load_model(model_path)
    cal_power(path1, IAF_band)
    # cal_power(path2, IAF_band)
    cal_power(path3, IAF_band)
    # cal_power(path4, IAF_band)