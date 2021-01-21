# -*- coding: utf-8 -*-
import wx
import os
import pickle
from BCIConfig import StimConfig, SubjectInfoConfig
from windows.TrainModelWindow import TrainModelWindow
from Pipeline import Pipeline
# from Exoskeleton import Exoskeleton


# 主窗体
class MainWindow(wx.Frame):
    def __init__(self):
        super(MainWindow, self).__init__(None, title="主界面", size=(290, 420))
        self.SetWindowStyle(wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
        self.init_param()
        self.init_ui()
        self.bind_event()

    def init_param(self):
        self.subject = SubjectInfoConfig()
        self.subjectNameList = os.listdir(self.subject.get_dataset_path())
        self.stim_cfg = StimConfig()
        # self.exo = Exoskeleton(self.subject)

    def init_ui(self):
        self.Centre()
        self.DestroyChildren()
        panel = wx.Panel(self)

        # wx.FlexGridSizer: 二维网状布局(rows, cols, vgap, hgap)=>(行数, 列数, 垂直方向行间距, 水平方向列间距)
        grid_sizer1 = wx.FlexGridSizer(cols=3, vgap=5, hgap=1)
        label = wx.StaticText(panel, label="被试：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.subjectNameCtrl = wx.Choice(panel, name="Subject Name", choices=self.subjectNameList, size=(100, 27))
        grid_sizer1.Add(self.subjectNameCtrl, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.newSubjectBtn = wx.Button(panel, label="新建被试", size=(90, 27))
        grid_sizer1.Add(self.newSubjectBtn, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)

        grid_sizer2 = wx.FlexGridSizer(cols=1, vgap=5, hgap=1)
        self.acqBtn = wx.Button(panel, label="① 校准(无反馈)", name="Acq", size=(110, 27))
        grid_sizer2.Add(self.acqBtn, 0, wx.ALL, 5)
        self.TrainModelBtn = wx.Button(panel, label="② 校准模型", size=(110, 27))
        grid_sizer2.Add(self.TrainModelBtn, 0, wx.ALL, 5)
        self.onlineBtn = wx.Button(panel, label="③ 校准(有反馈)", name="Online", size=(110, 27))
        grid_sizer2.Add(self.onlineBtn, 0, wx.ALL, 5)
        self.alpha_NFBtn = wx.Button(panel, label="④ alpha 有反馈训练", name="RestNF", size=(150, 27))
        grid_sizer2.Add(self.alpha_NFBtn, 0, wx.ALL, 5)
        self.ERD_NFBtn = wx.Button(panel, label="⑤ ERD 有反馈训练", name="LRNF", size=(150, 27))
        grid_sizer2.Add(self.ERD_NFBtn, 0, wx.ALL, 5)
        self.alpha_nonNFBtn = wx.Button(panel, label="④ alpha 无反馈训练", name="Rest_nonNF", size=(150, 27))
        grid_sizer2.Add(self.alpha_nonNFBtn, 0, wx.ALL, 5)
        self.ERD_nonNFBtn = wx.Button(panel, label="⑤ ERD 无反馈训练", name="LR_nonNF", size=(150, 27))
        grid_sizer2.Add(self.ERD_nonNFBtn, 0, wx.ALL, 5)

        self.statusBar = self.CreateStatusBar()  # 状态栏
        self.statusBar.SetStatusText(u'……')

        gridSizer = wx.FlexGridSizer(cols=1, vgap=1, hgap=1)
        gridSizer.Add(grid_sizer1, 0, wx.ALL, 5)
        gridSizer.Add(grid_sizer2, 0, wx.ALL, 5)

        panel.SetSizerAndFit(gridSizer)
        panel.Center()
        self.Fit()

    def bind_event(self):
        # Bind: 响应button事件
        self.newSubjectBtn.Bind(wx.EVT_BUTTON, self.on_new_subject)
        self.subjectNameCtrl.Bind(wx.EVT_CHOICE, self.on_load_param)
        self.acqBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
        self.onlineBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
        self.alpha_NFBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
        self.ERD_NFBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
        self.alpha_nonNFBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
        self.ERD_nonNFBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
        self.TrainModelBtn.Bind(wx.EVT_BUTTON, self.on_train_model)

    def on_new_subject(self, event):
        # 新建被试
        new_subject_dlg = wx.TextEntryDialog(self, '输入新被试名：', '新建被试')
        if new_subject_dlg.ShowModal() == wx.ID_OK:
            new_subject_name = new_subject_dlg.GetValue()
            self.subject.set_subject(new_subject_name)
            self.subjectNameList.append(new_subject_name)
            self.subjectNameCtrl.SetItems(self.subjectNameList)
            self.subjectNameCtrl.SetStringSelection(new_subject_name)
        new_subject_dlg.Destroy()

    def on_load_param(self, event):
        subject_name = self.subjectNameCtrl.GetStringSelection()
        self.subject.set_subject(subject_name)
        self.subject.set_date_dir()

    def on_graz_start(self, event):
        if not self.subject.subject_name:
            self.statusBar.SetStatusText(r'未选择被试')
            return
        self.session_type = event.GetEventObject().GetName()
        task_label = event.GetEventObject().GetLabel()
        # if self.session_type == 'Online' and not os.path.exists(self.subject.get_model_path()):
        #     self.statusBar.SetStatusText(r'未找到训练模型')
        #     return
        # self.subject.set_date_dir()

        msg_dialog = wx.MessageDialog(self, "是否开始【" + task_label + "】任务?", task_label+"任务开始", wx.OK | wx.CANCEL | wx.CENTRE)
        if msg_dialog.ShowModal() == wx.ID_OK:
            # self.exo.is_feedback = self.is_feedback
            pipline = Pipeline(self)
            pipline.start()
        else:
            return

    def on_train_model(self, event):
        if self.subject.subject_name:
            train_model_win = TrainModelWindow(self, "模型训练")
            train_model_win.ShowModal()
        else:
            self.statusBar.SetStatusText(r'未选择被试')
