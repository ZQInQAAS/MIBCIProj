import numpy as np
from time import strftime, sleep
from BCIConfig import BCIEvent
from Processor import Processor
from Stimulator import Stimulator
from NSDataReader import NSDataReaderRandom
from Interface_canvas import Interface


class Pipeline(object):
    def __init__(self, main_cfg):
        self.main_cfg = main_cfg
        # self.ns_reader = NSDataReader()
        self.ns_reader = NSDataReaderRandom()
        self.interface = Interface(main_cfg)
        self.stim_cfg = main_cfg.stim_cfg
        self.stim = Stimulator(main_cfg)
        self.save_data_path = main_cfg.subject.get_date_dir()
        self.is_feedback = True if self.main_cfg.session_type in ['online', 'RestNF', 'LRNF'] else False
        self.subscribe()

    def subscribe(self):
        self.stim.subscribe(BCIEvent.change_stim, self.interface.handle_stim)
        # self.stim.subscribe(BCIEvent.change_stim, main_cfg.exo.handle_stim)
        # self.cue.subscribe(BCIEvent.gaze_focus, self.stim.get_gaze)
        self.interface.subscribe(BCIEvent.cue_disconnect, self.stim.stop_stim)
        self.stim.subscribe(BCIEvent.stim_stop, self.save_data)
        self.stim.subscribe(BCIEvent.stim_stop, self.ns_reader.stop_data_reader)
        if self.is_feedback:
            self.processor = Processor(self.main_cfg)
            self.processor.subscribe(BCIEvent.readns_header, self.ns_reader.get_head_settings)
            self.processor.subscribe(BCIEvent.readns, self.ns_reader.get_ns_signal)
            self.stim.subscribe(BCIEvent.change_stim, self.processor.handle_stim)
            self.processor.subscribe(BCIEvent.online_bar, self.interface.online_bar)
            self.processor.subscribe(BCIEvent.online_face, self.interface.online_face)
            # self.processor.subscribe(BCIEvent.online_ctrl, main_cfg.exo.online_feedback)
            self.stim.subscribe(BCIEvent.stim_stop, self.processor.save_log)

    def start(self):
        self.ns_reader.start()
        sleep(2)
        self.stim.start()
        if self.is_feedback:
            self.processor.start()
        self.interface.Show()

    def save_data(self):
        nsheader_dict = self.ns_reader.get_head_settings()
        ns_signal = self.ns_reader.get_ns_signal()
        events, stim_log = self.stim.get_stimdata(self.ns_reader.data_time, ns_signal.shape[0])
        event_id_dict = self.main_cfg.stim_cfg.get_class_dict()
        stim_pram_dict = self.main_cfg.stim_cfg.get_stim_pram()
        path = strftime(self.save_data_path + "//" + self.main_cfg.session_type + "_%Y%m%d_%H%M_%S")
        np.savez(path, signal=ns_signal, events=events, stim_log=stim_log)
        path_config = strftime(self.save_data_path + "//" + 'config_%Y%m%d')
        np.savez(path_config, event_id_dict=event_id_dict, nsheader_dict=nsheader_dict, stim_pram_dict=stim_pram_dict)
        print('Signal data saved successfully.')


if __name__ == '__main__':
    np.load(r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean'
            r'\S4\S4_20200722\NSsignal_2020_07_22_16_22_30.npz')