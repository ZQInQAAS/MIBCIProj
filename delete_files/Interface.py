import os
import wx
import time
import random
import subprocess
from pubsub import pub
from process_tools import PyPublisher
from threading import Thread, Event
from BCIConfig import BCIEvent, StimType


class Interface(PyPublisher, wx.Frame):
    def __init__(self, main_cfg):
        super().__init__()
        super(PyPublisher, self).__init__(None, title="MIBCI", size=(1200, 960))
        # self.publish(BCIEvent.stim_stop)
        self.main_cfg = main_cfg
        self.stim = None
        self.isMax = False
        self.SetBackgroundColour('Black')
        pub.subscribe(self.update_display, 'update')
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        size = self.GetClientSize()
        self.buffer = wx.Bitmap(size.width, size.height)  # 创建一张空白位图作为缓冲区
        # self.buffer2 = wx.Bitmap(size.width, size.height)
        # imgpath = "cue_material/"
        # self.images = [os.path.join(imgpath, img) for img in os.listdir(imgpath) if img.lower().endswith('.png')]

    def clear_buffer(self):
        dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()

    def OnSize(self, event):
        self.window_size = self.GetClientSize()  # width, height
        self.centerPoint = wx.Point((self.window_size / 2).Get())
        self.buffer = wx.Bitmap(self.window_size.width, self.window_size.height)
        # print(self.window_size, self.centerPoint)

    def handle_stim(self, stim):
        self.stim = stim
        wx.CallAfter(pub.sendMessage, "update", msg=None)

    def update_display(self, msg):
        if self.stim in [StimType.ExperimentStart, StimType.EndOfTrial]:
            self.clear_buffer()
            return
        bias = 0  # 图片显示位置偏移
        if self.stim == StimType.CrossOnScreen:
            path = r'../cue_material/cross_white.png'
        elif self.stim == StimType.Rest:
            path = r'../cue_material/cross_blue.png'
        elif self.stim == StimType.Left:
            path = r'../cue_material/left_hand.png'
            bias = -400
        elif self.stim == StimType.Right:
            path = r'../cue_material/right_hand.png'
            bias = 400
        else:
            return
        # w = random.randint(1, 7)
        # path = self.images[w]
        bmp_image = wx.Image(path, wx.BITMAP_TYPE_PNG)
        # bmp_image = image.Scale(200, 120)
        bmp = wx.Bitmap(bmp_image)
        bsize = bmp_image.GetSize()
        xpos = (self.window_size.width - bsize[0]) / 2 + bias
        ypos = (self.window_size.height - bsize[1]) / 2
        dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
        dc.DrawBitmap(bmp, xpos, ypos)

    def plot_rect(self, score, color):
        print(color)
        dc = wx.ClientDC(self)
        # score = random.randint(20, 70)
        if isinstance(score, tuple):
            dc.DrawRectangle(self.centerPoint.x-75-50, self.centerPoint.y, 50, score[0]*10)  # x, y, width, height
            dc.DrawRectangle(self.centerPoint.x+75, self.centerPoint.y, 50, score[1]*10)
        else:
            dc.DrawRectangle(self.centerPoint.x-25, self.centerPoint.y, 50, score*10)

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        if key == wx.WXK_SPACE and self.isMax is False:  # 按下空格键 全屏
            c_x, c_y, c_w, c_h = wx.ClientDisplayRect()  # 取得桌面显示区域高
            e_width = 1680
            e_height = 1050
            self.SetSize(wx.Size(e_width, e_height))
            self.SetPosition(wx.Point(0, 0))  # c_w, 0

    def onClose(self, event):
        print('Interface closed.')

    def online_bar(self, score):
        print(self.stim, score)
        if isinstance(score, tuple):
            a, b = score
            difference = abs(a - b)
        else:
            rest = score
            difference = score
        color = 'Green' if difference > 5 else 'Yellow'
        self.plot_rect(score, color)


    def online_face(self):
        pass

if __name__ == '__main__':
    app = wx.App()
    win = Interface('1')
    win.Show()
    app.MainLoop()
