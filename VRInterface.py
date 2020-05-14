import json
import zmq
import subprocess
import datetime
from CueInterface import Interface
from BCIEnum import StimType, BCIEvent


class VRInterface(Interface):
    def __init__(self, main_cfg):
        super(VRInterface, self).__init__(main_cfg)
        self.gesture = main_cfg.subject.exo_type
        self.highpoint, _ = main_cfg.subject.get_exo_position()
        print(self.gesture, self.highpoint)
        self.PUB_address = 'tcp://*:12345'
        self.REP_address = 'tcp://*:12346'
        self.pname = 'C:\\Users\\HEDY\\Desktop\\hand_VR\\Unity3D FPS Handy Hands.exe'
        # self.pname = 'C:\\Users\\HEDY\\Desktop\\word_VR\\Unity3D FPS Handy Hands.exe'

    def start(self):
        subprocess.Popen(self.pname)
        context = zmq.Context()
        self.pub = context.socket(zmq.PUB)
        self.rep = context.socket(zmq.REP)
        self.pub.bind(self.PUB_address)
        self.rep.bind(self.REP_address)
        while True:
            recv_msg = self.rep.recv_string()
            if recv_msg == 'connect request':
                print(recv_msg)
                self.rep.send_string('OK')
                break

    def handle_stim(self, stim):
        if stim in self.class_list:
            self.class_name = stim
            return
        if stim == StimType.ExperimentStart:
            self.stim_to_message('ExpStart', '', '', False)
            self.send_message({'gesture': self.gesture})
            self.send_message({'highpoint': self.highpoint})
            return
        if stim == StimType.StartOfTrial:
            self.stim_to_message('StartTrial', '', '', False)
            return
        if stim == StimType.MoveUp:
            self.stim_to_message('MoveUp', self.class_name, 'move', self.is_online)
            if self.class_name == 'left':
                with open('C:\\Users\\HEDY\\Desktop\\python_timedata.txt', 'a') as file:
                    timedata = datetime.datetime.now().strftime('%H:%M:%S.%f')
                    file.writelines(timedata + '\n')
                    file.close()
            # print(datetime.datetime.now())  #向上运动的时候读取时间
            return
        if stim == StimType.MoveDown:
            self.stim_to_message('MoveDown', self.class_name, '', self.is_online)
            return
        if stim == StimType.Still:
            self.stim_to_message('Still', '', 'relax', self.is_online)
            return
        if stim == StimType.EyeGaze:
            self.send_focus_request(self.gaze_pos[self.class_name])
            return
        if stim == StimType.EndOfTrial:
            self.stim_to_message('EndTrial', self.class_name, 'stop', False)
            return
        if stim == StimType.ExperimentStop:
            self.stim_to_message('ExpStop', '', '', False)
            self.rep.close()
            self.pub.close()
            return

    def stim_to_message(self, stimtype, side, music, online):
        message = {'Type': stimtype, 'Side': side, 'Music': music, 'Online': online}
        self.send_message(message)

    def send_message(self, message):
        myjson = json.dumps(message)
        self.pub.send_string(myjson)

    def send_focus_request(self, gaze_position):
        pass

    def send_animation_ctrl(self, is_stop):
        self.send_message({'Type': 'animation_ctrl', 'Stop': is_stop, 'Online': True})

    def send_progress(self, progress):
        self.send_message({'Type': 'progress', 'Progress': progress})

    def online_feedback(self, predict):
        self.send_animation_ctrl(is_stop=bool(1-predict))
