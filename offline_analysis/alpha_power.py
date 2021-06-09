import os
import numpy as np
from BCIConfig import pick_rest_ch
from MIdataset_NF import MIdataset, cal_power_feature

def load_model(path):
    baseline_model = dict(np.load(path, allow_pickle=True))
    IAF_band = baseline_model['alpha_band']
    PAF = baseline_model['PAF']
    print('PAF', PAF)
    return IAF_band

def cal_power(path, AlphaBand):
    eo_baseline_pre = dict(np.load(path, allow_pickle=True))
    baseline_eo = eo_baseline_pre['signal']
    power, rela_power = cal_power_feature(baseline_eo, pick_rest_ch, freq_min=AlphaBand[0], freq_max=AlphaBand[1], rp=True)
    print('power:', round(power, 3), 'rp:', round(rela_power, 3))
    return power, rela_power


if __name__ == '__main__':
    p = os.path.abspath(os.path.dirname(os.getcwd()))
    p = p + r'\data_set\CYH\CYH_20210607'
    path1 = p + r'\Baseline_pre_eo.npz'
    path2 = p + r'\Baseline_pre_ec.npz'
    path3 = p + r'\Baseline_post_eo.npz'
    path4 = p + r'\Baseline_post_ec.npz'
    model_path = p + r'\model.npz'
    IAF_band = load_model(model_path)
    cal_power(path1, IAF_band)
    cal_power(path2, IAF_band)
    cal_power(path3, IAF_band)
    cal_power(path4, IAF_band)