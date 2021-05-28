import wx
import time
import numpy as np
# import subprocess
# from pubsub import pub
from process_tools import PyPublisher
# from threading import Thread, Event
from BCIConfig import BCIEvent, StimType
from wx.lib.floatcanvas import FloatCanvas
from BCIConfig import MRcorrAns1, MRcorrAns2


class Interface(PyPublisher, wx.Frame):
    def __init__(self, main_cfg):
        super().__init__()
        super(PyPublisher, self).__init__(None, title="MIBCI", size=(1200, 960))

        # self.main_cfg = main_cfg
        self.session_type = main_cfg.session_type
        self.save_path = main_cfg.subject.get_date_dir()
        self.NF_time_len = main_cfg.stim_cfg.NF_training_duration
        self.init_data()
        self.Bind(wx.EVT_CHAR_HOOK, self.onKey)
        self.rect0 = FloatCanvas.Rectangle((0, 0), (0, 0), FillColor='Red')
        # self.rect1 = FloatCanvas.Rectangle((0, 0), (0, 0), FillColor='Red')
        self.rect_t = FloatCanvas.Rectangle((0, 0), (0, 0), FillColor='Red')  # 进度条
        self.Canvas = FloatCanvas.FloatCanvas(self, -1, size=(1200, 960), ProjectionFun=None,
                                              Debug=0, BackgroundColor="Black", )

    def init_data(self):
        self.stim = None
        self.t0 = 0
        self.face_num = 0
        self.e_width = 1680
        self.e_height = 1050
        self.bar_len = 0  # 当前反馈bar的长度
        self.q_idx = 1  # MR问题的序号
        self.is_answer = False
        self.MR_answer = set()
        self.MR_answer_list = []
        self.MR_t0 = 0
        self.MR_tlist = []
        self.istimefeedback = False

    def handle_stim(self, stim):
        self.stim = stim
        if stim == StimType.Statement:
            path = r'cue_material/statement.png'
            self.draw_img(path, (0, 0))
        elif stim in [StimType.ExperimentStart, StimType.EndOfTrial, StimType.EndOfBaseline]:
            self.clear()
        elif stim == StimType.CrossOnScreen:
            path = r'cue_material/cross_white.png'
            self.draw_img(path, (0, 0))
        elif stim == StimType.Rest:
            path = r'cue_material/cross_green.png'
            self.draw_img(path, (0, 0))
            # self.is_rest = True
        elif stim == StimType.Left:
            path = r'cue_material/left_hand.png'
            self.draw_img(path, (-400, 0))
        elif stim == StimType.Right:
            path = r'cue_material/right_hand.png'
            self.draw_img(path, (400, 0))
        elif stim == StimType.LRCue:
            # self.is_rest = False
            path = r'cue_material/left_hand.png'
            self.draw_img(path, (-400, 0))
            path = r'cue_material/right_hand.png'
            self.draw_img(path, (400, 0))
        elif stim in [StimType.LRNF, StimType.RestNF]:
            self.t0 = time.time()
            self.Canvas.AddObject(self.rect0)
            self.Canvas.AddObject(self.rect_t)
            # if stim == StimType.LRNF:
                # self.Canvas.AddObject(self.rect1)
        elif stim == StimType.StartOfMR:
            MRpart_idx = 1 if self.session_type == 'MRPre' else 2
            path = r'cue_material/MRT/P' + str(MRpart_idx) + 'Q' + str(self.q_idx) + '.png'
            self.q_idx = self.q_idx + 1
            self.draw_img(path, (0, 100), height=300)
            # path = r'cue_material/MRT/MRT_option.png'
            # self.draw_img(path, (210, -90))
        elif stim == StimType.AnswerOfMR:
            self.is_answer = True
            path = r'cue_material/MRT/MRT_answer.png'
            self.draw_img(path, (-600, -140))
            self.MR_t0 = time.time()
        elif stim == StimType.ExperimentStop:
            if self.MR_answer_list:
                self.saveMR()
            self.clear()
            self.Destroy()
        else:
            return

    def draw_img(self, path, xy, height=None):
        image = wx.Image(path)
        if height is None:
            img = FloatCanvas.Bitmap(image, xy, Position='cc')
        else:
            img = FloatCanvas.ScaledBitmap(image, xy, Height=height, Position='cc')
        self.Canvas.AddObject(img)
        self.Canvas.Draw()

    def clear(self):
        self.MR_answer = set()
        self.is_answer = False
        self.bar_len = 0
        self.Canvas.ClearAll()
        time.sleep(0.1)
        self.Canvas.Draw()
        # print('clear.')

    def onClose(self, event):
        self.publish(BCIEvent.cue_disconnect)
        print('Interface closed.')

    def onKey(self, event):
        if not self.is_answer:
            return
        if event.GetKeyCode() == wx.WXK_NUMPAD1:
            if len(self.MR_answer) < 2:
                self.MR_answer.add(1)
                path = r'cue_material/MRT/MR_choice.png'
                self.draw_img(path, (-200, -150))
            # print("1 key pressed")
        elif event.GetKeyCode() == wx.WXK_NUMPAD2:
            if len(self.MR_answer) < 2:
                self.MR_answer.add(2)
                path = r'cue_material/MRT/MR_choice.png'
                self.draw_img(path, (50, -150))
            # print("2 key pressed")
        elif event.GetKeyCode() == wx.WXK_NUMPAD3:
            if len(self.MR_answer) < 2:
                self.MR_answer.add(3)
                path = r'cue_material/MRT/MR_choice.png'
                self.draw_img(path, (300, -150))
            # print("3 key pressed")
        elif event.GetKeyCode() == wx.WXK_NUMPAD4:
            if len(self.MR_answer) < 2:
                self.MR_answer.add(4)
                path = r'cue_material/MRT/MR_choice.png'
                self.draw_img(path, (550, -150))
            # print("4 key pressed")
        elif event.GetKeyCode() == wx.WXK_NUMPAD_SUBTRACT:
            self.MR_answer = set()
            path = r'cue_material/MRT/MR_hidden.png'
            self.draw_img(path, (210, -160))
            # print("SUBTRACT key pressed")
        elif event.GetKeyCode() == wx.WXK_NUMPAD_ENTER:
            self.MR_tlist.append(time.time() - self.MR_t0)
            self.MR_answer_list.append(self.MR_answer)
            self.MR_answer = set()
            path = r'cue_material/MRT/MR_submit.png'
            self.draw_img(path, (0, -350))
            time.sleep(1)
            self.publish(BCIEvent.MRsubmit)  # MR等提交后trial结束
            print("ENTER key pressed. Answer is ", self.MR_answer)
        else:
            event.Skip()

    def saveMR(self):
        # score = 1  # TODO:MR分数计算
        # corrAns = MRcorrAns1 if self.session_type == 'MRPre' else MRcorrAns2
        np.savez(self.save_path + r'/' + self.session_type + r'_result',
                 MR_answer=self.MR_answer_list, tlist=self.MR_tlist)
        print(self.session_type + ' results saved.')

    def online_bar(self, score, label, is_reached):
        # print(time.time(), score, is_reached)
        bar_width = 50
        self.bar_len = self.bar_len + score * 10  # TODO 调整速度参数
        # bar_len = score * 150
        color_name = 'orange' if self.bar_len > 0 and label in [StimType.Rest, StimType.Right] or \
                                 (self.bar_len < 0 and label == StimType.Left) else 'slate blue'
        # color_name = 'red' if is_reached else color_name
        if label == StimType.Rest:
            self.set_rect_param(self.rect0, x=-bar_width / 2, y=0, w=bar_width, h=self.bar_len, color=color_name)  # 上下
        else:
            self.set_rect_param(self.rect0, x=0, y=-bar_width / 2, w=self.bar_len, h=bar_width, color=color_name)  # 左右
            # self.set_rect_param(self.rect1, x=0, y=0, w=bar_width, h=score[1] * 150, color=color)
            # self.Canvas.RemoveObject(self.rect0)
            # self.Canvas.AddObject(self.rect0)
        t = time.time() - self.t0
        if 1000 > t > 6:
            bar_width = t * 1400 / self.NF_time_len
            self.set_rect_param(self.rect_t, x=-700, y=-500, w=bar_width, h=10, color='grey')  # 时间进度条
        self.Canvas.Draw(Force=True)

    def set_rect_param(self, rect, x, y, w, h, color):
        rect.SetShape((x, y), (w, h))  # ((x,y), (w,h))
        rect.SetFillColor(color)
        rect.SetLineColor(color)

    def online_face(self, label):
        self.bar_len = 0
        print('face', self.face_num)
        if label == StimType.Rest:
            num_each_line = 10
            facepath_id = self.face_num // num_each_line + 1  # 颜色序号
            face_pos = self.face_num % num_each_line + 1  # 第N个
            xy = (-600 + face_pos * 100, 450)  # (0,0)
        else:
            num_each_line = 6
            num_oneside = self.face_num // 2
            facepath_id = num_oneside // num_each_line + 1  # 颜色序号
            face_pos = num_oneside % num_each_line + 1  # 第N个
            isleft = -1 if (self.face_num % 2) == 1 else 1  # 奇数是左，偶数是右
            xy = (800 * isleft, 500 - face_pos * 100)
        facepath_id = facepath_id if facepath_id <= 4 else (facepath_id - 4)
        path = r'cue_material/smiley' + str(facepath_id) + '.png'
        self.draw_img(path, xy)
        self.face_num = self.face_num + 1


def cal_MR_score(MR_answer, session_type):
    score = 0
    corrAns = MRcorrAns1 if session_type == 'MRPre' else MRcorrAns2
    for i in range(len(corrAns)):
        if corrAns[i] == tuple(MR_answer[i]):
            score = score + 1
    return score


if __name__ == '__main__':
    # app = wx.App()
    # win = Interface('1')
    # win.Show()
    # app.MainLoop()
    MRPre_result = dict(np.load(r'data_set\S4\S4_20210527\MRPre_result.npz', allow_pickle=True))
    MR_answer = MRPre_result['MR_answer']
    tlist = MRPre_result['tlist']
    s = cal_MR_score(MR_answer, 'MRPre')
    print(s)
