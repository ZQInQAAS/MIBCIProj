import os
import json
import socket
import subprocess
from threading import Thread, Lock
from BCIEnum import BCIEvent, StimType


class CueInterface(object):
    def __init__(self, pybus, main_cfg):
        self.state = None
        self.pybus = pybus
        self.is_online = main_cfg.is_online
        self.sock = socket.socket()
        self.tcp_lock = Lock()
        self.tcp_address = "127.0.0.1", 4567
        self.tcp_recv_thread = Thread(target=self.tcp_recv)
        self.class_list = main_cfg.stim_cfg.class_list
        self.cue_path = main_cfg.stim_cfg.cue_path
        self.is_repeat = main_cfg.stim_cfg.is_repeat
        self.move_sound_path = main_cfg.stim_cfg.move_sound_path
        self.relax_sound_path = main_cfg.stim_cfg.relax_sound_path
        self.stop_sound_path = main_cfg.stim_cfg.stop_sound_path
        # self.parentDir = os.path.abspath(os.getcwd())
        self.parentDir = os.getcwd()
        self.gaze_pos = {'left': 'Left', 'right': 'Right', 'rest': 'Center'}

    def start(self):
        self.connect()
        self.tcp_recv_thread.start()

    def connect(self):
        self.sock.bind(self.tcp_address)
        self.sock.listen(1)
        parent_dir = os.path.abspath(os.getcwd())
        subprocess.Popen(parent_dir + '\\Release\\SharpBCI.exe ' +
                         parent_dir + '\\Release\\Config\\MI_cursor.scfg')
        self.conn, _ = self.sock.accept()

    def handle_stim(self, stim):
        if stim in self.class_list:
            self.class_name = stim
            self.class_idx = self.class_list.index(stim)
            return
        if stim == StimType.ExperimentStart:
            self.send_test('准备')
            return
        if stim == StimType.MoveUp:
            cue_path = self.cue_path[self.class_idx][0] if self.is_repeat else self.cue_path[self.class_idx]
            self.send_stim(cue_path, self.move_sound_path, self.is_online)
            return
        if stim == StimType.MoveDown:
            self.send_stim(self.cue_path[self.class_idx][1], None, self.is_online)
            return
        if stim == StimType.Still:
            self.send_stim(self.cue_path[self.class_idx], self.relax_sound_path, False)
            return
        if stim == StimType.EyeGaze:
            self.send_clear()
            self.send_focus_request(self.gaze_pos[self.class_name])
            return
        if stim == StimType.StartOfTrial:
            self.send_clear()
            self.predict_one_trial = []
            return
        if stim == StimType.EndOfTrial:
            self.send_stim(None, self.stop_sound_path, False)
            return
        if stim == StimType.ExperimentStop:
            self.send_end()
            self.close()

    def tcp_recv(self):
        while True:
            try:
                lengthBytes = bytearray(4)
                self.conn.recvfrom_into(lengthBytes, 4)
                length = int.from_bytes(lengthBytes, byteorder='big')
                content, _ = self.conn.recvfrom(length)
                message = json.loads(content)
                if message['Type'] == 'focused':
                    self.pybus.publish(self, BCIEvent.gaze_focus)  # Resume stim_cfg
                elif message['Type'] == 'closed':
                    self.disconnect()
            except:
                continue

    def online_feedback(self, predict):
        self.send_animation_ctrl(is_stop=bool(1-predict))

    def disconnect(self):
        self.pybus.publish(self, BCIEvent.cue_disconnect)
        self.close()

    def close(self):
        self.conn.close()
        self.sock.close()
        print('Interface closed.')

    def send_message(self, message):
        # print(message)
        json_bytes = json.dumps(message).encode(encoding="utf-8")
        self.tcp_lock.acquire()
        try:
            self.conn.sendall(bytes((1, 3)))
            self.conn.sendall((len(json_bytes)).to_bytes(4, byteorder='big'))
            self.conn.sendall(json_bytes)
        except socket.error:
            print("cue tcp connection lost...")
        self.tcp_lock.release()

    def send_stim(self, visual_resource, auditory_resource, is_progress_bar, duration=0):
        if visual_resource:
            visual_resource = 'image:file://' + self.parentDir + visual_resource
        if auditory_resource:
            auditory_resource = 'audio:file://' + self.parentDir + auditory_resource
        self.send_message({'Type': 'stage', 'VisualResource': visual_resource, 'AuditoryResource': auditory_resource,
                           'Duration': duration, 'ProgressBar': is_progress_bar})

    def send_test(self, text):
        text = 'text:' + text
        self.send_message({'Type': 'stage', 'VisualResource': text, 'AuditoryResource': None,
                           'Duration': 0, 'ProgressBar': False})

    def send_end(self):
        self.send_message({'Type': 'stage', 'End': True})

    def send_clear(self):
        self.send_stim(None, None, False)
        self.send_progress(0)

    def send_focus_request(self, gaze_position):
        self.send_message({'Type': 'focus', 'GazePosition': gaze_position})

    def send_progress(self, progress):
        self.send_message({'Type': 'progress', 'Progress': progress})

    def send_animation_ctrl(self, is_stop):
        self.send_message({'Type': 'animation_ctrl', 'IsStopCtrl': is_stop})

