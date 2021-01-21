import wx
import time
# import subprocess
# from pubsub import pub
from process_tools import PyPublisher
# from threading import Thread, Event
from BCIConfig import BCIEvent, StimType
from wx.lib.floatcanvas import FloatCanvas


class Interface(PyPublisher, wx.Frame):
    def __init__(self, main_cfg):
        super().__init__()
        super(PyPublisher, self).__init__(None, title="MIBCI", size=(1200, 960))
        # self.publish(BCIEvent.stim_stop)
        self.main_cfg = main_cfg
        self.NF_time_len = main_cfg.stim_cfg.NF_training_duration
        self.stim = None
        self.isMax = False
        self.face_num = 0
        self.e_width = 1680
        self.e_height = 1050
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.rect0 = FloatCanvas.Rectangle((-125, 0), (0, 0), FillColor='Red')
        self.rect1 = FloatCanvas.Rectangle((-125, 0), (0, 0), FillColor='Red')
        self.rect_t = FloatCanvas.Rectangle((-125, 0), (0, 0), FillColor='Red')
        self.Canvas = FloatCanvas.FloatCanvas(self, -1, size=(1200, 960), ProjectionFun=None,
                                              Debug=0, BackgroundColor="Black", )

    def handle_stim(self, stim):
        self.stim = stim
        if stim == StimType.ExperimentStart:
            self.clear()
            self.Canvas.AddObject(self.rect0)
            self.Canvas.AddObject(self.rect1)
            self.Canvas.AddObject(self.rect_t)
            return
        if stim == StimType.EndOfTrial:
            self.clear()
            return
        elif stim == StimType.CrossOnScreen:
            path = r'cue_material/cross_white.png'
            self.draw_img(path, (0, 0))
        elif stim == StimType.Rest:
            path = r'cue_material/cross_blue.png'
            self.draw_img(path, (0, 0))
            self.is_rest = True
            self.t0 = time.time()
        elif stim == StimType.Left:
            path = r'cue_material/left_hand.png'
            self.draw_img(path, (-400, 0))
        elif stim == StimType.Right:
            path = r'cue_material/right_hand.png'
            self.draw_img(path, (400, 0))
        elif stim == StimType.LRCue:
            self.t0 = time.time()
            self.is_rest = False
            path = r'cue_material/left_hand.png'
            self.draw_img(path, (-400, 0))
            path = r'cue_material/right_hand.png'
            self.draw_img(path, (400, 0))
        elif stim in [StimType.LRNF, StimType.RestNF] and (time.time() - self.t0) > 5:
            self.istimefeedback = True
        if stim == StimType.ExperimentStop:
            self.clear()
            self.Destroy()
        else:
            return

    def draw_img(self, path, xy):
        image = wx.Image(path)
        img = FloatCanvas.Bitmap(image, xy, Position='cc')
        self.Canvas.AddObject(img)
        self.Canvas.Draw()

    def clear(self):
        self.Canvas.ClearAll()
        self.Canvas.Draw()

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        if key == wx.WXK_SPACE and self.isMax is False:  # 按下空格键 全屏
            print('max')
            # c_x, c_y, c_w, c_h = wx.ClientDisplayRect()  # 取得桌面显示区域高
            self.SetSize(wx.Size(self.e_width, self.e_height))
            self.SetPosition(wx.Point(0, 0))  # c_w, 0

    def onClose(self, event):
        print('Interface closed.')

    def online_bar(self, score, is_reached):
        color = 'Green' if is_reached else 'Yellow'
        print(score[1]-score[0], color)
        bar_width = 50
        bar_bias = 100
        if isinstance(score, tuple):
            self.rect0.SetShape((-bar_bias - bar_width, 0), (bar_width, score[0] * 150))  # ((x,y), (w,h))
            self.rect0.SetFillColor(color)
            self.rect0.SetLineColor(color)
            self.rect1.SetShape((bar_bias, 0), (bar_width, score[1] * 150))
            self.rect1.SetFillColor(color)
            self.rect1.SetLineColor(color)
            self.Canvas.RemoveObject(self.rect0)
            self.Canvas.AddObject(self.rect0)
            self.Canvas.RemoveObject(self.rect1)
            self.Canvas.AddObject(self.rect1)
        else:
            self.rect0.SetShape((-bar_width / 2, 0), (bar_width, score * 150))
            self.rect0.SetFillColor(color)
            self.Canvas.RemoveObject(self.rect0)
            self.Canvas.AddObject(self.rect0)
        if self.istimefeedback:
            bar_width = (time.time() - self.t0) * 1200 / self.NF_time_len
            self.rect_t.SetShape((-700, -600), (bar_width, 50))  # ((x,y), (w,h))
            self.rect_t.SetFillColor('grey')
            self.rect_t.SetLineColor('grey')
            self.Canvas.RemoveObject(self.rect_t)
            self.Canvas.AddObject(self.rect_t)
        self.Canvas.Draw()

    def online_face(self):
        print('face', self.face_num)
        if self.is_rest:
            num_each_line = 10
            facepath_id = self.face_num // num_each_line + 1  # 颜色序号
            face_pos = self.face_num % num_each_line + 1  # 第N个
            facepath_id = facepath_id if facepath_id <= 4 else (facepath_id - 4)
            path = r'cue_material/smiley' + str(facepath_id) + '.png'
            self.draw_img(path, (-600 + face_pos * 100, 450))  # (0,0)
        else:
            num_each_line = 6
            num_oneside = self.face_num // 2
            facepath_id = num_oneside // num_each_line + 1  # 颜色序号
            facepath_id = facepath_id if facepath_id <= 4 else (facepath_id - 4)
            path = r'cue_material/smiley' + str(facepath_id) + '.png'
            face_pos = num_oneside % num_each_line + 1  # 第N个
            isleft = -1 if (self.face_num % 2) == 1 else 1  # 奇数是左，偶数是右
            self.draw_img(path, (700 * isleft, 500 - face_pos * 100))
        self.face_num = self.face_num + 1




if __name__ == '__main__':
    app = wx.App()
    win = Interface('1')
    win.Show()
    app.MainLoop()