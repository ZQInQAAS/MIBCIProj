import mne
import time
import random
import numpy as np
import pandas as pd
from queue import Queue
from time import strftime
from scipy.integrate import simps
from process_tools import PyPublisher
from NSDataReader import RepeatingTimer
from BCIConfig import BCIEvent, StimType, ch_types, ch_names

rest_ch = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1',
           'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']


class Processor(PyPublisher):
    def __init__(self, main_cfg):
        PyPublisher.__init__(self)
        self.class_list = main_cfg.stim_cfg.class_list
        self.model_path = main_cfg.subject.get_model_path()
        self.save_path = main_cfg.subject.get_date_dir()
        self.epoch_dur = 0.1  # 每多少秒判断一次
        self.proc_bar_len = main_cfg.stim_cfg.display_cue_duration/self.epoch_dur
        self.online_timer = RepeatingTimer(self.epoch_dur, self.online_run)
        self.trial_num = 0
        self.right_num_one_run = []
        self.right_num_all = []
        self.result_log = []
        self.init_data()

    def init_data(self):
        self.predict_state = False
        self.left_threshold = 0.1
        self.right_threshold = 0.1
        self.rest_threshold = 0.1
        self.wait_list_maxlen = int(5 / self.epoch_dur)  # 5s为一个保持周期
        self.wait_list = []
        self.rela_left_power_list = []
        self.rela_right_power_list = []
        self.rela_rest_power_list = []
        self.left_threshold_list = []
        self.right_threshold_list = []
        self.rest_threshold_list = []
        self.t_stride = 0.02  # 阈值调整步长
        self.is_left = None

    def start(self):
        selected_ch_pd = pd.read_csv(self.model_path, header=None)
        self.left_ch = list(selected_ch_pd.iloc[:, 0].values)  #eval(selected_ch_pd.iloc[0].values[0])
        self.right_ch = list(selected_ch_pd.iloc[:, 1].values)  #eval(selected_ch_pd.iloc[1].values[0])
        self.fs = self.publish(BCIEvent.readns_header)
        self.online_timer.start()
        self.info = mne.create_info(ch_names, self.fs, ch_types)
        self.info.set_montage('standard_1005')

    def handle_stim(self, stim):
        print(time.time(), stim)
        if stim in [StimType.Left, StimType.Right, StimType.Rest]:
            self.label, self.label_idx = stim.name, stim.value
        if stim == StimType.LRCue:
            self.label, self.label_idx = 'Right', stim.value  # 右先
        elif stim in [StimType.LRNF, StimType.RestNF]:
            self.wait_list = []
            self.t0 = time.time()
            self.predict_state = True
        elif stim == StimType.EndOfBaseline:
            self.baseline_signal = self.publish(BCIEvent.readns, duration=45*self.fs)  # 45s signals
            self.base_power_alpha = self.cal_power_feature(self.baseline_signal, rest_ch, fmin=8, fmax=13)
            self.base_power_ERDleft = self.cal_power_feature(self.baseline_signal, self.left_ch, fmin=8, fmax=30)
            self.base_power_ERDright = self.cal_power_feature(self.baseline_signal, self.right_ch, fmin=8, fmax=30)
        elif stim == StimType.EndOfTrial:
            self.predict_state = False
            self.get_result_log()
        elif stim == StimType.ExperimentStop:
            print('stop processor')
            self.predict_state = False
            self.online_timer.stop()
        else:
            return

    def online_run(self):
        if self.predict_state:
            signal = self.publish(BCIEvent.readns, duration=500)  # signal (sample, channal)
            rela_power, is_reached = self.is_reached_threshold(signal)
            self.publish(BCIEvent.online_bar, rela_power, is_reached)
            self.wait_list.append(is_reached)
            if len(self.wait_list) == self.wait_list_maxlen:
                if sum(self.wait_list) > len(self.wait_list)*0.65:  # sum(self.wait_list) == len(self.wait_list)
                    self.publish(BCIEvent.online_face)
                    self.up_threshold()
                    if self.label == 'Left':  # 切换另一侧 MI
                        self.label = 'Right'
                    elif self.label == 'Right':
                        self.label = 'Left'
                    self.wait_list = []
                elif sum(self.wait_list) < len(self.wait_list)*0.35:  # sum(self.wait_list) == 0
                    self.down_threshold()
                if self.wait_list != []:
                    self.wait_list.pop(0)

    def up_threshold(self):
        if self.label == 'Left':
            left_power, right_power = self.avg_power()
            erd = right_power - left_power
            self.left_threshold, is_up = (self.left_threshold, False) \
                if self.left_threshold > (erd - self.t_stride) else (erd - self.t_stride, True)
            if is_up:
                self.left_threshold_list.append([time.time() - self.t0, self.left_threshold])
                print('Up threshold. Left_threshold:', self.left_threshold)
        elif self.label == 'Right':
            left_power, right_power = self.avg_power()
            erd = left_power - right_power
            self.right_threshold, is_up = (self.right_threshold, False) \
                if self.right_threshold > (erd - self.t_stride) else (erd - self.t_stride, True)
            if is_up:
                self.right_threshold_list.append([time.time() - self.t0, self.right_threshold])
                print('Up threshold. Right_threshold:', self.right_threshold)
        else:  # Rest
            rela_rest_power = np.mean(self.rela_rest_power_list[-self.wait_list_maxlen:-1])
            self.rest_threshold, is_up = (self.rest_threshold, False) \
                if self.rest_threshold > (rela_rest_power - self.t_stride) else (rela_rest_power - self.t_stride, True)
            if is_up:
                self.rest_threshold_list.append([time.time() - self.t0, self.rest_threshold])
                print('Up threshold, Rest_threshold', self.rest_threshold)

    def avg_power(self):
        left = np.mean(self.rela_left_power_list[-self.wait_list_maxlen:-1])
        right = np.mean(self.rela_right_power_list[-self.wait_list_maxlen:-1])
        return left, right

    def down_threshold(self):
        if self.label == 'Left':
            left_power, right_power = self.avg_power()
            erd = right_power - left_power
            self.left_threshold, is_down = (self.left_threshold, False) \
                if self.left_threshold < (erd + self.t_stride) else (erd + self.t_stride, True)
            if is_down:
                self.left_threshold_list.append([time.time() - self.t0, self.left_threshold])
                print('Down threshold, Left_threshold', self.left_threshold)
        elif self.label == 'Right':
            left_power, right_power = self.avg_power()
            erd = left_power - right_power
            self.right_threshold, is_down = (self.right_threshold, False) \
                if self.right_threshold < (erd + self.t_stride) else (erd + self.t_stride, True)
            if is_down:
                self.right_threshold_list.append([time.time() - self.t0, self.right_threshold])
                print('Down threshold, Right_threshold', self.right_threshold)
        else:  # Rest
            r_rest_power = np.mean(self.rela_rest_power_list[-self.wait_list_maxlen:-1])
            self.rest_threshold, is_down = (self.rest_threshold, False) \
                if self.rest_threshold < (r_rest_power + self.t_stride) else (r_rest_power + self.t_stride, True)
            if is_down:
                self.rest_threshold_list.append([time.time() - self.t0, self.rest_threshold])
                print('Down threshold, Rest_threshold', self.rest_threshold)

    def is_reached_threshold(self, signal):
        # signal (sample, channal)
        if self.label in ['Left', 'Right']:
            left_power = self.cal_power_feature(signal, self.left_ch, fmin=8, fmax=30)
            right_power = self.cal_power_feature(signal, self.right_ch, fmin=8, fmax=30)
            rela_left_power = (self.base_power_ERDleft - left_power) / self.base_power_ERDleft
            rela_right_power = (self.base_power_ERDright - right_power) / self.base_power_ERDright
            self.rela_left_power_list.append(rela_left_power)
            self.rela_right_power_list.append(rela_right_power)
            if self.label == 'Left':
                erd = rela_right_power - rela_left_power
                return (rela_left_power, rela_right_power), erd > self.left_threshold
            else:
                erd = rela_left_power - rela_right_power
                return (rela_left_power, rela_right_power), erd > self.right_threshold
        else:  # Rest
            rest_power = self.cal_power_feature(signal, rest_ch, fmin=8, fmax=13)
            rela_rest_power = (rest_power - self.base_power_alpha) / self.base_power_alpha
            self.rela_rest_power_list.append(rela_rest_power)
            return rela_rest_power, rela_rest_power > self.rest_threshold

    def cal_power_feature(self, signal, ch, fmin=8, fmax=30):
        raw_mne = mne.io.RawArray(signal.T, self.info, verbose='warning')  # RawArray input (n_channels, n_times)
        raw_mne.set_eeg_reference(ref_channels='average', projection=True, verbose='warning').apply_proj()  # CAR
        # raw_mne.filter(8, 30, verbose='warning')  # band pass
        raw_mne = raw_mne.pick_channels(ch)
        raw_mne = raw_mne.reorder_channels(ch)
        psd, freqs = mne.time_frequency.psd_welch(raw_mne, fmin=fmin, fmax=fmax, proj=True, verbose='warning')  # psd (channal, freq)
        freq_res = freqs[1] - freqs[0]  # 频率分辨率
        power_ch = simps(psd, dx=freq_res)  # 求积分 power:(channal,)
        power = np.average(power_ch)  # 通道平均
        return power

    def get_result_log(self):

        self.trial_num = self.trial_num + 1
        # print('————————————')
        one_run_log = '\ntrial ' + str(self.trial_num) + ':  ' + 'cue:' + str(self.label)
                      # '\nthis run acc:' + str(one_run_acc) + ', epoch num:' + str(epoch_num) + \
                      # '\nall runs acc:' + str(all_acc)
        self.result_log.append(one_run_log)
        # print(one_run_log)

    def save_log(self):
        if self.label == 'Rest' and self.rest_threshold_list != []:
            rela_rest_pd = pd.DataFrame(data=self.rela_rest_power_list)
            rest_threshold_pd = pd.DataFrame(data=self.rest_threshold_list)
            rela_rest_pd.to_csv(self.save_path + r'/log/relative_rest_power.csv', header=False, index=False)
            rest_threshold_pd.to_csv(self.save_path + r'/log/rest_threshold.csv', header=False, index=False)
        elif self.right_threshold_list != [] or self.left_threshold_list != []:
            rela_right_pd = pd.DataFrame(data=self.rela_right_power_list)
            right_threshold_pd = pd.DataFrame(data=self.right_threshold_list)
            rela_right_pd.to_csv(self.save_path + r'/log/relative_right_power.csv', header=False, index=False)
            right_threshold_pd.to_csv(self.save_path + r'/log/right_threshold.csv', header=False, index=False)
            rela_left_pd = pd.DataFrame(data=self.rela_left_power_list)
            left_threshold_pd = pd.DataFrame(data=self.left_threshold_list)
            rela_left_pd.to_csv(self.save_path + r'/log/relative_left_power.csv', header=False, index=False)
            left_threshold_pd.to_csv(self.save_path + r'/log/left_threshold.csv', header=False, index=False)
        else:
            print('error log saved')
        print('Online log saved successfully.')


if __name__ == '__main__':
    from MIdataset import MIdataset
    dataset_path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean'
    data_path = r'\S4\S4_20200721\NSsignal_2020_07_21_16_15_11.npz'
    data = MIdataset(dataset_path + data_path)
    data.set_reference()
    signal = data.get_raw_data()
    signal1 = signal[6000:6500, :]
    p = Processor('1')
    p.label = 'Left'
    p.fs = 500
    p.left_ch = ['F3', 'F1', 'C2', 'C4', 'C6']
    p.right_ch = ['FC2', 'FC4', 'FC5', 'FC3', 'FC1']
    p.ch_names = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1',
                  'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'F5', 'AF3', 'AF4', 'P5',
                  'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PO3', 'POz', 'PO4', 'Oz', 'F6']  # 41ch EEG
    p.ch_types = ['eeg'] * 41
    p.info = mne.create_info(p.ch_names, p.fs, p.ch_types)
    p.info.set_montage('standard_1005')
    p.fs = 500
    p.baseline_signal = signal[7000:9500, :]
    # p.base_power_alpha = p.cal_power_feature(p.baseline_signal, rest_ch, fmin=8, fmax=13)
    p.base_power_ERDleft = p.cal_power_feature(p.baseline_signal, p.left_ch, fmin=8, fmax=30)
    p.base_power_ERDright = p.cal_power_feature(p.baseline_signal, p.right_ch, fmin=8, fmax=30)
    is_reached = p.is_reached_threshold(signal1)