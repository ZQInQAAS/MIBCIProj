import wx
import os
import pickle
import numpy as np
from process_tools.load_data import loadnpz
from process_tools import Classification
import pandas as pd
from MIdataset_NF import MIdataset, cal_power_feature
from process_tools import iterative_CSP_LR
from BCIConfig import pick_motor_ch, pick_rest_ch


class TrainModelWindow(wx.Dialog):
    def __init__(self, parent, title):
        super(TrainModelWindow, self).__init__(parent, title=title)
        # self.save_model_path = parent.subject.get_model_path()
        self.save_model_path = parent.subject.get_date_dir()
        # self.train_file_num = parent.trainFileNumCtrl.GetValue()
        self.init_ui()

    def init_ui(self):
        dataWildcard = "npz Data File (.npz)" + "|*.npz"
        panel = wx.Panel(self)
        grid_sizer1 = wx.FlexGridSizer(cols=2, vgap=1, hgap=1)
        label_text = '选择数据' + '：'
        label = wx.StaticText(panel, label=label_text)
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.train_path_ctrl = wx.FilePickerCtrl(panel, wildcard=dataWildcard, size=(350, 27))
        self.train_path_ctrl.SetInitialDirectory(self.save_model_path)
        self.train_path_ctrl.GetPickerCtrl().SetLabel('浏览')
        grid_sizer1.Add(self.train_path_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.SetSize(470, 60 * 2 + 30)
        self.Centre()
        grid_sizer2 = wx.FlexGridSizer(cols=2, vgap=1, hgap=1)
        self.train_model_btn = wx.Button(panel, label='模型训练开始', size=wx.Size(100, 27))
        self.train_model_btn.Bind(wx.EVT_BUTTON, self.on_train_model)
        grid_sizer2.Add(self.train_model_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.statusLabel = wx.StaticText(panel, label=' ')
        grid_sizer2.Add(self.statusLabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        grid_sizer = wx.FlexGridSizer(cols=1, vgap=1, hgap=1)
        grid_sizer.Add(grid_sizer1, 0, wx.ALL, 5)
        grid_sizer.Add(grid_sizer2, 0, wx.ALL, 5)
        panel.SetSizerAndFit(grid_sizer)
        panel.Center()
        self.Fit()

    def on_close(self, event):
        self.Close()  # 关闭窗体

    def get_UA_RP(self, baseline_eo):
        # get peak alpha frequency(paf) from eye_closed baseline
        ec_baseline_pre = MIdataset(self.save_model_path + r'/Baseline_pre_ec.npz')
        ec_baseline_pre.bandpass_filter(1, 100)  # band pass
        PAF, AlphaBand = ec_baseline_pre.get_IAF(fmin=7, fmax=14)
        print('PAF:', PAF, 'band:', AlphaBand)
        # PAF=10
        alpha_band = (PAF-2, PAF+2)
        base_UA_power, base_rela_UA_power = cal_power_feature(baseline_eo, pick_rest_ch, freq_min=alpha_band[0],
                                                              freq_max=alpha_band[1], rp=True)
        return PAF, alpha_band, base_UA_power, base_rela_UA_power

    def get_individual_LR(self, baseline_eo):
        data = MIdataset(self.train_path_ctrl.GetPath())
        # data.bandpass_filter(1, 100)  # band pass
        # data.set_reference()  # CAR
        data.bandpass_filter(8, 30)
        select_ch = pick_motor_ch  # M1附近区域
        data_array, label = data.get_epoch_data(select_label=['Left', 'Right'], select_ch=select_ch)
        selected_ch_names = iterative_CSP_LR(data_array, select_ch)  # csp select
        left_ch = selected_ch_names[:, 0]
        right_ch = selected_ch_names[:, 1]
        base_leftch_power = cal_power_feature(baseline_eo, left_ch, freq_min=8, freq_max=30)
        base_rightch_power = cal_power_feature(baseline_eo, right_ch, freq_min=8, freq_max=30)
        return left_ch, right_ch, base_leftch_power, base_rightch_power

    def on_train_model(self, event):
        eo_baseline_pre = dict(np.load(self.save_model_path + r'/Baseline_pre_eo.npz', allow_pickle=True))
        baseline_eo = eo_baseline_pre['signal']
        PAF, alpha_band, UApower, UA_RP = self.get_UA_RP(baseline_eo)
        # left_ch, right_ch, base_leftch_power, base_rightch_power = self.get_individual_LR(baseline_eo)
        left_ch, right_ch, base_leftch_power, base_rightch_power = None, None, None, None
        np.savez(self.save_model_path + r'/model', left_ch=left_ch, right_ch=right_ch, PAF=PAF, alpha_band=alpha_band,
                 base_alpha_power=UApower, base_alpha_rela_power=UA_RP,
                 base_leftch_power=base_leftch_power, base_rightch_power=base_rightch_power)
        # df.to_csv(self.save_model_path, header=False, index=False)  # save selected channels
        self.statusLabel.SetLabel('模型训练完成。')


if __name__ == '__main__':
    # model = dict(np.load(r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\S1\S1_20210528\model.npz'))
    # base_alpha_rela_power = model['base_alpha_rela_power']
    pa = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\LCY\LCY_20210601\Acq_20210601_1502_08.npz'
    import os
    from matplotlib import pyplot as plt
    # p = os.path.abspath(os.path.dirname(os.getcwd()))
    p0= r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\WRQ\WRQ_20210615\Baseline_pre_ec.npz'
    p1 = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\PNM\PNM_20210607\Baseline_pre_ec.npz'
    p2 = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\CYH\CYH_20210607\Baseline_post_ec.npz'
    p3 = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\XY\XY_20210607\Baseline_post_ec.npz'
    ec_baseline_pre = MIdataset(p2)
    # ec_baseline_pre.bandpass_filter(1, 100)  # band pass
    fig, ax = plt.subplots()
    PAF, AlphaBand = ec_baseline_pre.get_IAF(fmin=7.5, fmax=13.5, ax=ax)
    plt.show()
    print('PAF:', PAF, 'band:', AlphaBand)