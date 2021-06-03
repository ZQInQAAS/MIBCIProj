import os
import numpy as np
from time import strftime, sleep
from BCIConfig import BCIEvent, fs
from Processor import Processor
from Stimulator import Stimulator
from NSDataReader import NSDataReaderRandom, NSDataReader
from Interface_canvas import Interface


class Pipeline(object):
    def __init__(self, main_cfg):
        self.main_cfg = main_cfg
        self.ns_reader = NSDataReader()
        # self.ns_reader = NSDataReaderRandom()
        self.interface = Interface(main_cfg)
        self.stim_cfg = main_cfg.stim_cfg
        self.stim = Stimulator(main_cfg)
        self.save_data_path = main_cfg.subject.get_date_dir()
        self.is_feedback = True if self.main_cfg.session_type in ['Online', 'RestNF', 'LRNF'] else False
        self.subscribe()

    def subscribe(self):
        self.stim.subscribe(BCIEvent.change_stim, self.interface.handle_stim)
        # self.stim.subscribe(BCIEvent.change_stim, main_cfg.exo.handle_stim)
        # self.interface.subscribe(BCIEvent.gaze_focus, self.stim.get_gaze)
        self.stim.subscribe(BCIEvent.stim_stop, self.save_data)
        self.stim.subscribe(BCIEvent.stim_stop, self.ns_reader.stop_data_reader)
        self.interface.subscribe(BCIEvent.MRsubmit, self.stim.MRsubmit)
        self.interface.subscribe(BCIEvent.cue_disconnect, self.stim.stop_stim)
        # self.interface.subscribe(BCIEvent.cue_disconnect, self.ns_reader.stop_data_reader)
        if self.is_feedback:
            self.processor = Processor(self.main_cfg)
            self.processor.subscribe(BCIEvent.readns_header, self.ns_reader.get_sample_rate)
            self.processor.subscribe(BCIEvent.readns, self.ns_reader.get_ns_signal)
            self.stim.subscribe(BCIEvent.change_stim, self.processor.handle_stim)
            self.processor.subscribe(BCIEvent.online_bar, self.interface.online_bar)
            self.processor.subscribe(BCIEvent.online_face, self.interface.online_face)
            # self.processor.subscribe(BCIEvent.online_ctrl, main_cfg.exo.online_feedback)
            self.stim.subscribe(BCIEvent.stim_stop, self.processor.stop)

    def start(self):
        self.ns_reader.start()
        sleep(2)
        self.stim.start()
        if self.is_feedback:
            self.processor.start()
        self.interface.Show()

    def save_data(self):
        # nsheader_dict = self.ns_reader.get_head_settings()
        ns_signal = self.ns_reader.get_ns_signal()
        ns_signal = np.array(ns_signal)
        events, stim_log = self.stim.get_stimdata(self.ns_reader.data_time, ns_signal.shape[0])
        event_id_dict = self.main_cfg.stim_cfg.get_class_dict()
        # stim_pram_dict = self.main_cfg.stim_cfg.get_stim_pram()
        is_pre = '_pre' if self.main_cfg.is_pre else '_post'
        if self.main_cfg.session_type == 'Baseline':
            ns_signal = self.cal_baseline(ns_signal, stim_log)
            path_name = self.save_data_path + r'/' + self.main_cfg.session_type + is_pre
            np.savez(path_name + '_eo', signal=ns_signal[0], events=events, stim_log=stim_log)
            np.savez(path_name + '_ec', signal=ns_signal[1], events=events, stim_log=stim_log)
        else:
            if self.main_cfg.session_type == 'MRT':
                path = self.save_data_path + r"/" + self.main_cfg.session_type + is_pre
            elif self.main_cfg.session_type == 'Acq':
                path = strftime(self.save_data_path + r'/' + self.main_cfg.session_type + is_pre + "_%Y%m%d_%H%M_%S")
            else:  # NF
                path = strftime(self.save_data_path + r"/" + self.main_cfg.session_type + "_%Y%m%d_%H%M_%S")
            np.savez(path, signal=ns_signal, events=events, stim_log=stim_log)
        # path_config = strftime(self.save_data_path + "//" + 'config_%Y%m%d')
        # np.savez(path_config, event_id_dict=event_id_dict, sample_rate=nsheader_dict['sample_rate'],
        #          ch_names=nsheader_dict['ch_names'], ch_types=nsheader_dict['ch_types'],
        #          stim_pram_dict=stim_pram_dict)
        print('Signal data saved successfully.')

    def cal_baseline(self, ns_signal, stim_log):
        first_time, last_time = self.ns_reader.data_time[0], self.ns_reader.data_time[-1]
        data_time = np.linspace(0, last_time - first_time, num=ns_signal.shape[0])
        base_idx = [stim_log.iloc[2, 0], stim_log.iloc[3, 0],  # idx of ['CrossOnScreen', 'EndOfBaseline'] eye-open
                    stim_log.iloc[4, 0], stim_log.iloc[5, 0]]  # eye-closed
        fs = 200  # NS random时
        for i in range(len(base_idx)):
            for j in range(int(base_idx[i] * fs - 50), len(data_time)):
                if data_time[j] > base_idx[i]:
                    v1 = abs(data_time[j - 1] - base_idx[i])
                    v2 = abs(data_time[j] - base_idx[i])
                    base_idx[i] = j - 1 if v1 < v2 else j
                    break
        ns_signal_eo = ns_signal[base_idx[0]:base_idx[1], :]
        ns_signal_ec = ns_signal[base_idx[2]:base_idx[3], :]
        return ns_signal_eo, ns_signal_ec


if __name__ == '__main__':
    # np.load(r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean'
            # r'\S4\S4_20200722\NSsignal_2020_07_22_16_22_30.npz')
    path = r'data_set\S4\S4_20210428\baseline_pre.npz'
    npz_data_dict = dict(np.load(path, allow_pickle=True))
    a = npz_data_dict['signal']
