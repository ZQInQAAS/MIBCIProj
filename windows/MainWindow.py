# -*- coding: utf-8 -*-
import wx
import os
import pickle
import datetime
from SubjectInfoConfig import SubjectInfoConfig
from StimConfig import StimConfig
from windows.SettingWindow import SettingWindow
from windows.TrainModelWindow import TrainModelWindow
from windows.ExoskeletonWindow import ExoskeletonWindow
from Pipeline import Pipeline
from Exoskeleton import Exoskeleton


# 主窗体
class MainWindow(wx.Frame):
    def __init__(self):
        super(MainWindow, self).__init__(None, title="主界面", size=(320, 490))
        self.SetWindowStyle(wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
        self.init_param()
        self.init_ui()
        self.bind_event()

    def init_param(self):
        self.subject = SubjectInfoConfig()
        self.subjectNameList = os.listdir(self.subject.get_dataset_path())
        self.stim_cfg = StimConfig()
        self.exo = Exoskeleton(self.subject)

    def init_ui(self):
        self.Centre()
        self.DestroyChildren()
        panel = wx.Panel(self)

        title_img_dir = os.path.abspath(os.getcwd()) + r'\cue_material\title_img.png'
        image = wx.Image(title_img_dir, wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        title_bitmap = wx.StaticBitmap(panel, -1, image)

        # wx.FlexGridSizer: 二维网状布局(rows, cols, vgap, hgap)=>(行数, 列数, 垂直方向行间距, 水平方向列间距)
        grid_sizer1 = wx.FlexGridSizer(cols=3, vgap=5, hgap=1)
        label = wx.StaticText(panel, label="被试：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.subjectNameCtrl = wx.Choice(panel, name="Subject Name", choices=self.subjectNameList, size=(100, 27))
        grid_sizer1.Add(self.subjectNameCtrl, 0, wx.ALL, 5)
        self.saveParamBtn = wx.Button(panel, label="保存参数", size=(100, 27))
        grid_sizer1.Add(self.saveParamBtn, 0, wx.ALL, 5)

        label = wx.StaticText(panel, label="")
        grid_sizer1.Add(label, 0, wx.ALL, 5)
        self.acqBtn = wx.Button(panel, label="① 校准", name="acquisition", size=(100, 27))
        grid_sizer1.Add(self.acqBtn, 0, wx.ALL, 5)
        self.onlineBtn = wx.Button(panel, label="③ 在线训练", name="online", size=(100, 27))
        grid_sizer1.Add(self.onlineBtn, 0, wx.ALL, 5)

        label = wx.StaticText(panel, label="训练文件：")
        grid_sizer1.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.trainFileNumCtrl = wx.SpinCtrl(panel, value='1', min=1, max=6, size=(80, 27))
        grid_sizer1.Add(self.trainFileNumCtrl, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.TrainModelBtn = wx.Button(panel, label="② 模型训练", size=(100, 27))
        grid_sizer1.Add(self.TrainModelBtn, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)

        sbox = wx.StaticBox(panel, -1, label=u'参数配置')
        sbsizer = wx.StaticBoxSizer(sbox, wx.VERTICAL)

        grid_sizer2 = wx.FlexGridSizer(cols=3, vgap=10, hgap=1)
        self.newSubjectBtn = wx.Button(panel, label="新建被试", size=(120, 27))
        grid_sizer2.Add(self.newSubjectBtn, 0, wx.ALL, 5)
        label = wx.StaticText(panel, label="患侧：")
        grid_sizer2.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.affectedSideCtrl = wx.Choice(panel, name="Affected Side", choices=['左', '右'], size=(70, 27))
        self.affectedSideCtrl.SetSelection(0)
        grid_sizer2.Add(self.affectedSideCtrl, 0, wx.ALL, 5)

        grid_sizer3 = wx.FlexGridSizer(cols=2, vgap=10, hgap=1)
        self.settingExoBtn = wx.Button(panel, label="外骨骼配置", size=(120, 27))
        grid_sizer3.Add(self.settingExoBtn, 0, wx.ALL, 5)
        self.settingCueBtn = wx.Button(panel, label="提示界面配置", size=(120, 27))
        # self.settingCueBtn.Disable()  # 禁用btn
        grid_sizer3.Add(self.settingCueBtn, 0, wx.ALL, 5)

        sbsizer.Add(grid_sizer2, proportion=0, flag=wx.ALL, border=5)
        sbsizer.Add(grid_sizer3, proportion=0, flag=wx.ALL, border=5)

        self.statusBar = self.CreateStatusBar()  # 状态栏
        self.statusBar.SetStatusText(u'……')

        gridSizer = wx.FlexGridSizer(cols=1, vgap=1, hgap=1)
        gridSizer.Add(title_bitmap, 0, wx.ALL, 5)
        gridSizer.Add(grid_sizer1, 0, wx.ALL, 5)
        gridSizer.Add(sbsizer, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)

        panel.SetSizerAndFit(gridSizer)
        panel.Center()
        self.Fit()

    def bind_event(self):
        # Bind: 响应button事件
        self.newSubjectBtn.Bind(wx.EVT_BUTTON, self.on_new_subject)
        self.settingCueBtn.Bind(wx.EVT_BUTTON, self.on_cue_settings)
        self.settingExoBtn.Bind(wx.EVT_BUTTON, self.on_exo_settings)
        self.subjectNameCtrl.Bind(wx.EVT_CHOICE, self.on_load_param)
        self.affectedSideCtrl.Bind(wx.EVT_CHOICE, self.on_affected_side_change)
        self.saveParamBtn.Bind(wx.EVT_BUTTON, self.on_save_param)
        self.acqBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
        self.onlineBtn.Bind(wx.EVT_BUTTON, self.on_graz_start)
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

    def on_cue_settings(self, event):
        setting_win = SettingWindow(self, "自定义提示界面参数")
        setting_win.ShowModal()

    def on_exo_settings(self, event):
        exo_window = ExoskeletonWindow(self, "外骨骼配置")
        exo_window.ShowModal()

    def on_affected_side_change(self, event):
        affected_side = self.affectedSideCtrl.GetStringSelection()
        self.subject.affected_side = 'left' if affected_side == '左' else 'right'

    def on_load_param(self, event):
        subject_name = self.subjectNameCtrl.GetStringSelection()
        self.subject.set_subject(subject_name)
        param_path = self.subject.get_param_path()
        if os.path.exists(param_path):
            if os.path.getsize(param_path) > 0:
                with open(param_path, 'rb') as f:
                    self.subject = pickle.load(f)
                    self.stim_cfg = pickle.load(f)
                self.statusBar.SetStatusText(r'参数载入成功')
            self.affectedSideCtrl.SetStringSelection('左' if self.subject.affected_side == 'left' else '右')
        else:
            self.statusBar.SetStatusText(r'未保存参数文件')

    def on_save_param(self, event):
        if self.subject.subject_name:
            with open(self.subject.get_param_path(), 'wb') as f:
                f.write(pickle.dumps(self.subject))
                f.write(pickle.dumps(self.stim_cfg))
            self.statusBar.SetStatusText(r'参数文件保存成功')
        else:
            self.statusBar.SetStatusText(r'未选择被试')

    def on_graz_start(self, event):
        if not self.subject.subject_name:
            self.statusBar.SetStatusText(r'未选择被试')
            return
        session_type = event.GetEventObject().GetName()
        if session_type == 'online' and not os.path.exists(self.subject.get_model_path()):
            self.statusBar.SetStatusText(r'未找到训练模型')
            return
        self.subject.set_date_dir()
        tag, self.is_online = ('校准', False) if session_type == 'acquisition' else ('训练', True)
        msg_dialog = wx.MessageDialog(self, "是否开始【" + tag + "任务】?", tag+"任务开始", wx.OK | wx.CANCEL | wx.CENTRE)
        if msg_dialog.ShowModal() == wx.ID_OK:
            self.exo.is_online = self.is_online
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
