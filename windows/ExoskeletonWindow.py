import wx

exo_type_dist = {'肘屈伸': 'elbow',
                 '小臂旋转': 'forearm',
                 '腕屈伸': 'wrist',
                 '手握拳': 'hand'}


class ExoskeletonWindow(wx.Dialog):
    def __init__(self, parent, title):
        super(ExoskeletonWindow, self).__init__(parent, title=title, size=(320, 300))
        self.exo = parent.exo
        self.subject = parent.subject
        self.init_ui()
        self.init_value()

    def init_ui(self):
        self.Centre()
        panel = wx.Panel(self)
        grid_sizer1 = wx.FlexGridSizer(cols=3, vgap=10, hgap=1)
        self.is_exo_fb_check = wx.CheckBox(panel, -1, "外骨骼反馈")
        grid_sizer1.Add(self.is_exo_fb_check, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        label = wx.StaticText(panel, label="| 端口号：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.comNum = wx.SpinCtrl(panel, value='0', min=0, max=10000, size=(70, 27))
        grid_sizer1.Add(self.comNum, 0, wx.ALL, 5)

        exo_type_list = list(exo_type_dist.keys())
        self.exo_type_Ctrl = wx.Choice(panel, name="exo type", choices=exo_type_list, size=(90, 27))
        self.exo_type_Ctrl.SetSelection(0)
        grid_sizer1.Add(self.exo_type_Ctrl, 0, wx.ALL, 5)

        grid_sizer2 = wx.FlexGridSizer(cols=4, vgap=10, hgap=1)
        self.lowlabel = wx.StaticText(panel, label="最低点：")
        grid_sizer2.Add(self.lowlabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.lowestPoint = wx.SpinCtrl(panel, value='60', min=0, max=3000, size=(60, 27))  # 外骨骼最低点
        grid_sizer2.Add(self.lowestPoint, 0, wx.ALL, 5)
        self.highlabel = wx.StaticText(panel, label="最高点：")
        grid_sizer2.Add(self.highlabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.highestPoint = wx.SpinCtrl(panel, value='0', min=0, max=3000, size=(60, 27))  # 外骨骼最高点
        grid_sizer2.Add(self.highestPoint, 0, wx.ALL, 5)

        label = wx.StaticText(panel, label="速度：")
        grid_sizer2.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.velocity = wx.SpinCtrl(panel, value='30', min=0, max=80, size=(60, 27))  # 外骨骼速度
        grid_sizer2.Add(self.velocity, 0, wx.ALL, 5)

        grid_sizer3 = wx.FlexGridSizer(cols=3, vgap=10, hgap=1)
        self.previewBtn = wx.Button(panel, label="预览", size=(100, 23))
        self.previewBtn.Bind(wx.EVT_BUTTON, self.on_preview)
        grid_sizer3.Add(self.previewBtn, 0, wx.ALL, 5)

        self.submitBtn = wx.Button(panel, label="提交", size=(100, 23))
        self.submitBtn.Bind(wx.EVT_BUTTON, self.on_submit)
        grid_sizer3.Add(self.submitBtn, 0, wx.ALL, 5)

        grid_sizer = wx.FlexGridSizer(cols=1, vgap=10, hgap=1)
        grid_sizer.Add(grid_sizer1, 0, wx.ALL, 5)
        grid_sizer.Add(grid_sizer2, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        grid_sizer.Add(grid_sizer3, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        panel.SetSizerAndFit(grid_sizer)
        panel.Center()
        self.Fit()

    def on_submit(self, event):
        self.get_value()
        if self.exo.Connected is True:
            self.exo.disconnect_com()
        self.Close()  # 关闭窗体

    def on_preview(self, event):
        self.get_value()
        if not self.exo.Connected:
            self.exo.connect(self.comNum.GetValue())
        self.exo.preview_step()
        self.exo.disconnect_com()

    def on_stop(self, event):
        # 结束外骨骼控制，关蓝牙
        if self.exo and self.exo.Connected:
                self.exo.disconnect_com()

    def get_value(self):
        self.subject.exo_type = exo_type_dist[self.exo_type_Ctrl.GetStringSelection()]
        self.exo.is_exo_feedback = self.is_exo_fb_check.GetValue()  # 是否外骨骼反馈
        self.subject.set_exo_position(self.highestPoint.GetValue(), self.lowestPoint.GetValue())
        self.subject.set_exo_velocity(self.velocity.GetValue())

    def init_value(self):
        self.comNum.SetValue(self.subject.get_com_num())
        self.is_exo_fb_check.SetValue(self.exo.is_exo_feedback)
        high, low = self.subject.get_exo_position()
        self.highestPoint.SetValue(high)
        self.lowestPoint.SetValue(low)
        self.velocity.SetValue(self.subject.get_exo_velocity())
        self.exo_type_Ctrl.SetStringSelection(list(exo_type_dist.keys())[list(exo_type_dist.values()).index(self.subject.exo_type)])
        if self.subject.exo_type in ['forearm', 'wrist']:
            self.highlabel.SetLabel("最右点：")
            self.lowlabel.SetLabel("最左点：")
