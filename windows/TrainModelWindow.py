import wx
import os
import pickle
import numpy as np
from process_tools.load_data import loadnpz
from process_tools import Classification


class TrainModelWindow(wx.Dialog):
    def __init__(self, parent, title):
        super(TrainModelWindow, self).__init__(parent, title=title)
        self.save_model_path = parent.subject.get_model_path()
        self.train_file_num = parent.trainFileNumCtrl.GetValue()
        self.init_ui()

    def init_ui(self):
        dataWildcard = "npz Data File (.npz)" + "|*.npz"
        panel = wx.Panel(self)
        grid_sizer1 = wx.FlexGridSizer(cols=2, vgap=1, hgap=1)
        self.train_path_ctrl = list(range(self.train_file_num))
        for i in range(self.train_file_num):
            label_text = '选择数据' + str(i+1) + '：'
            label = wx.StaticText(panel, label=label_text)
            grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.train_path_ctrl[i] = wx.FilePickerCtrl(panel, wildcard=dataWildcard, size=(350, 27))
            self.train_path_ctrl[i].SetInitialDirectory(os.path.dirname(self.save_model_path))
            self.train_path_ctrl[i].GetPickerCtrl().SetLabel('浏览')
            grid_sizer1.Add(self.train_path_ctrl[i], 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.SetSize(470, 60 * (self.train_file_num + 1) + 30)
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

    def gather_data(self):
        data_num = len(self.train_path_ctrl)
        train_data_path = list(range(data_num))
        for i in range(data_num):
            train_data_path[i] = self.train_path_ctrl[i].GetPath()
            if train_data_path[i] == '':
                train_data_path.remove(train_data_path[i])
        train_data_path = list(set(train_data_path))
        data_x_gather, data_y_gather, fs = loadnpz(train_data_path[0])
        for i in range(1, len(train_data_path)):
            data_x, data_y, _ = loadnpz(train_data_path[i])  # x:(sample, channal, trial)  y:(trial,)
            data_x_gather = np.concatenate((data_x_gather, data_x), axis=2)
            data_y_gather = np.concatenate((data_y_gather, data_y), axis=0)
        return data_x_gather, data_y_gather, fs

    def on_train_model(self, event):
        data_x, data_y, fs = self.gather_data()
        clf = Classification()
        clf.train_model(data_x, data_y, fs)
        with open(self.save_model_path, 'wb') as f:
            f.write(pickle.dumps(clf))
        self.statusLabel.SetLabel('模型训练完成。')





