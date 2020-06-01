import wx
from BCIConfig import classStimList


class SettingWindow(wx.Dialog):
    def __init__(self, parent, title):
        super(SettingWindow, self).__init__(parent, title=title, size=(380, 360))
        self.stim_cfg = parent.stim_cfg
        self.init_ui()
        self.init_data()

    def init_ui(self):
        self.Centre()
        panel = wx.Panel(self)
        self.event_cb_list = list()

        grid_sizer0 = wx.FlexGridSizer(cols=3, vgap=10, hgap=1)
        for i in range(len(classStimList)):
            self.event_cb_list.append(wx.CheckBox(panel, label=classStimList[i]))
            self.event_cb_list[i].SetValue(classStimList[i] in self.stim_cfg.class_list)
            grid_sizer0.Add(self.event_cb_list[i], 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        grid_sizer1 = wx.FlexGridSizer(cols=2, vgap=10, hgap=1)
        self.oneClassNumCtrl = wx.SpinCtrl(panel, value='10', min=0, max=50)
        label = wx.StaticText(panel, label="每类任务次数：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        grid_sizer1.Add(self.oneClassNumCtrl, 0, wx.ALL, 5)

        self.baselineCtrl = wx.SpinCtrl(panel, value='5', min=0, max=20)
        label = wx.StaticText(panel, label="准备时长（秒）：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        grid_sizer1.Add(self.baselineCtrl, 0, wx.ALL, 5)

        self.waitCueCtrl = wx.SpinCtrl(panel, value='2', min=0, max=10)
        label = wx.StaticText(panel, label="提示间隔（秒）：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        grid_sizer1.Add(self.waitCueCtrl, 0, wx.ALL, 5)

        self.dispCueCtrl = wx.SpinCtrl(panel, value='5', min=0, max=20)
        label = wx.StaticText(panel, label="任务持续（秒）：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        grid_sizer1.Add(self.dispCueCtrl, 0, wx.ALL, 5)

        self.is_repeat_check = wx.CheckBox(panel, -1, "双向运动")
        grid_sizer1.Add(self.is_repeat_check, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        self.submitBtn = wx.Button(panel, label="提交", size=(100, 27))
        self.submitBtn.Bind(wx.EVT_BUTTON, self.on_submit)
        grid_sizer1.Add(self.submitBtn, 0, wx.ALL, 5)

        grid_sizer = wx.FlexGridSizer(cols=1, vgap=1, hgap=1)
        grid_sizer.Add(grid_sizer0, 0, wx.ALL, 5)
        grid_sizer.Add(grid_sizer1, 0, wx.ALL, 5)
        panel.SetSizerAndFit(grid_sizer)
        panel.Center()
        self.Fit()

    def on_submit(self, event):
        self.get_value()
        self.Close()  # 关闭窗体

    # 从窗体获取数据
    def get_value(self):
        self.stim_cfg.each_class_num = self.oneClassNumCtrl.GetValue()
        self.stim_cfg.baseline_duration = self.baselineCtrl.GetValue()
        self.stim_cfg.cue_interval_duration = self.waitCueCtrl.GetValue()
        self.stim_cfg.display_cue_duration = self.dispCueCtrl.GetValue()
        self.stim_cfg.is_repeat = self.is_repeat_check.GetValue()
        stim_list = list()
        for i in range(len(classStimList)):
            if self.event_cb_list[i].GetValue():
                stim_list.append(self.event_cb_list[i].GetLabel())
        self.stim_cfg.set_class(stim_list)

    # 界面初始化
    def init_data(self):
        self.oneClassNumCtrl.SetValue(self.stim_cfg.each_class_num)
        self.baselineCtrl.SetValue(self.stim_cfg.baseline_duration)
        self.waitCueCtrl.SetValue(self.stim_cfg.cue_interval_duration)
        self.dispCueCtrl.SetValue(self.stim_cfg.display_cue_duration)
        self.is_repeat_check.SetValue(self.stim_cfg.is_repeat)

