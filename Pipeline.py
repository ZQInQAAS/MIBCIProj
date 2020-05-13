import numpy as np
import pandas as pd
from time import strftime, sleep
from BCIEnum import BCIEvent
from utils import PyEventBus
from Processor import Processor
from Stimulator import Stimulator
from NSDataReader import NSDataReader, NSDataReaderRandom
from CueInterface import CueInterface
from VRInterface import VRInterface


class Pipeline(object):
    def __init__(self, main_cfg):
        self.ns_reader = NSDataReader()
        # self.ns_reader = NSDataReaderRandom()
        self.is_online = main_cfg.is_online
        # self.cue = CueInterface(main_cfg)
        self.cue = VRInterface(main_cfg)
        self.stim = Stimulator(main_cfg.stim_cfg)
        self.save_data_path = main_cfg.subject.get_date_dir()
        self.filename = 'online' if main_cfg.is_online else 'acquire'
        self.stim_cfg = main_cfg.stim_cfg
        self.stim.subscribe(BCIEvent.change_stim, self.cue.handle_stim)
        self.stim.subscribe(BCIEvent.change_stim, main_cfg.exo.handle_stim)
        self.cue.subscribe(BCIEvent.gaze_focus, self.stim.get_gaze)
        self.cue.subscribe(BCIEvent.cue_disconnect, self.stim.stop_stim)
        self.stim.subscribe(BCIEvent.save_data, self.save_data)
        self.stim.subscribe(BCIEvent.save_data, self.ns_reader.stop_data_reader)
        if self.is_online:
            self.processor = Processor(main_cfg)
            self.processor.subscribe(BCIEvent.readns_header, self.ns_reader.get_head_settings)
            self.processor.subscribe(BCIEvent.readns, self.ns_reader.get_ns_signal)
            self.stim.subscribe(BCIEvent.change_stim, self.processor.handle_stim)
            self.processor.subscribe(BCIEvent.online_progressbar, self.cue.send_progress)
            self.processor.subscribe(BCIEvent.online_ctrl, self.cue.online_feedback)
            self.processor.subscribe(BCIEvent.online_ctrl, main_cfg.exo.online_feedback)
            self.stim.subscribe(BCIEvent.save_data, self.processor.save_log)

    def start(self):
        self.ns_reader.start()
        self.cue.start()
        sleep(2)
        self.stim.start()
        if self.is_online:
            self.processor.start()

    def save_data(self):
        header_dict = self.ns_reader.get_head_settings()
        ns_signal = self.ns_reader.get_ns_signal()
        data_time = self.ns_reader.data_time
        first_time, last_time = data_time[0], data_time[-1]
        data_time = np.linspace(0, last_time - first_time, ns_signal.shape[0])
        class_list = np.array(self.stim.class_list)
        class_list[:, 0] = class_list[:, 0] - first_time
        stim_log = pd.DataFrame(self.stim.stim_list)
        stim_log.iloc[:, 0] = stim_log.iloc[:, 0] - first_time
        event_id_dict = self.stim_cfg.get_class_dict()
        stim_pram_dict = self.stim_cfg.get_stim_pram()
        events = np.zeros([class_list.shape[0], 3], dtype=np.int)
        events[:, 2] = class_list[:, 1]
        k = 0
        for i in range(class_list.shape[0]):
            for j in range(k, len(data_time)):
                if data_time[j] > class_list[i, 0]:
                    v1 = abs(data_time[j - 1] - class_list[i, 0])
                    v2 = abs(data_time[j] - class_list[i, 0])
                    events[i, 0] = j - 1 if v1 < v2 else j
                    k = j - 1
                    break
        path = strftime(self.save_data_path + "//" + self.filename + "NS_%Y%m%d_%H%M_%S")
        np.savez(path, signal=ns_signal, events=events, event_id_dict=event_id_dict,
                header_dict=header_dict, stim_pram_dict=stim_pram_dict, stim_log=stim_log)
        print('Signal data saved successfully.')

