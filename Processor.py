import os
import sys
import mne
import time
import random
import numpy as np
import pandas as pd

from queue import Queue
from time import strftime
from datetime import datetime
from scipy.integrate import simps

from process_tools import PyPublisher
from NSDataReader import RepeatingTimer
from BCIConfig import BCIEvent, StimType
from MIdataset_NF import cal_power_feature
from BCIConfig import pick_rest_ch, fs


class Processor(PyPublisher):
    def __init__(self, main_cfg):
        PyPublisher.__init__(self)
        # self.model_path = main_cfg.subject.get_model_path()
        self.save_path = main_cfg.subject.get_date_dir()
        self.epoch_dur = 0.04  # 每0.04秒判断一次  25Hz
        self.proc_bar_len = main_cfg.stim_cfg.display_cue_duration/self.epoch_dur
        self.online_timer = RepeatingTimer(self.epoch_dur, self.online_run)
        self.init_data()
        self.trial_num = 0
        # self.right_num_one_run = []
        # self.right_num_all = []
        self.result_log = []
        self.beseline_dur = main_cfg.stim_cfg.baseline_duration

    def init_data(self):
        self.predict_state = False
        self.is_reached_buffer_len = int(2 / self.epoch_dur)  # 4s一周期(100个epoch)  2s 50个
        self.is_reached_buffer = []
        self.power_buffer_len = 3  #
        self.power_buffer = []
        self.power_win_width = 1*fs  # 计算power的窗宽 1s
        self.rela_left_power_list = []  # rela: (power-base)/base
        self.rela_right_power_list = []
        self.rela_rest_power_list = []
        self.left_threshold_list = []
        self.right_threshold_list = []
        self.rest_threshold_list = []
        self.left_threshold = -0.05
        self.right_threshold = 0.05
        self.rest_threshold = 0.05
        self.t_stride = 0.01  # 阈值调整步长
        self.is_left = None

    def start(self):
        baseline_model = dict(np.load(self.save_path + r'/model.npz', allow_pickle=True))
        self.left_ch = baseline_model['left_ch']
        self.right_ch = baseline_model['right_ch']
        self.IAF_band = baseline_model['alpha_band']
        self.base_alpha_power = baseline_model['base_alpha_rela_power']
        self.base_leftch_power = baseline_model['base_leftch_power']
        self.base_rightch_power = baseline_model['base_rightch_power']
        # self.fs = self.publish(BCIEvent.readns_header)
        self.online_timer.start()

    def stop(self):
        self.online_timer.cancel()
        self.save_log()
        print('stop processor')

    def handle_stim(self, stim):
        # print('processor', time.time(), stim)
        if stim in [StimType.Left, StimType.Right, StimType.Rest]:
            self.label = stim
        elif stim == StimType.LRCue:
            self.label = StimType.Right  # 右先
        elif stim in [StimType.LRNF, StimType.RestNF]:
            self.is_reached_buffer = []
            self.t0 = time.time()
            self.predict_state = True
        elif stim == StimType.EndOfTrial:
            self.predict_state = False
            self.get_result_log()
        elif stim == StimType.ExperimentStop:
            self.stop()
        else:
            return

    def online_run(self):
        if self.predict_state:
            signal = self.publish(BCIEvent.readns, duration=self.power_win_width)
            signal = np.array(signal)  # signal (sample, channal)
            rela_power, is_reached = self.is_reached_threshold(signal)
            self.power_buffer.append(rela_power)
            if len(self.power_buffer) > self.power_buffer_len:
                self.power_buffer.pop(0)
            # print(time.time(), 'processor')
            try:
                avg_power = np.mean(self.power_buffer)
                # avg_power = rela_power
                self.publish(BCIEvent.online_bar, avg_power, self.label, is_reached)
            except RuntimeError:
                print('Interface has been deleted.')
                sys.exit(1)
            self.is_reached_buffer.append(is_reached)
            print('wait_list:', len(self.is_reached_buffer))
            if len(self.is_reached_buffer) == self.is_reached_buffer_len:  # NF session
                if sum(self.is_reached_buffer) > len(self.is_reached_buffer) * 0.7:  # sum(self.wait_list) == len(self.wait_list)
                    self.publish(BCIEvent.online_face, self.label)
                    self.up_threshold()
                    self.is_reached_buffer = []
                    if self.label == StimType.Left:  # 切换另一侧 MI
                        self.label = StimType.Right
                    elif self.label == StimType.Right:
                        self.label = StimType.Left
                elif sum(self.is_reached_buffer) < len(self.is_reached_buffer)*0.3:  # sum(self.wait_list) == 0
                    self.down_threshold()
                if self.is_reached_buffer != []:
                    self.is_reached_buffer.pop(0)

    def is_reached_threshold(self, signal):
        # signal (sample, channal)
        if self.label == StimType.Rest:
            rest_power, rest_power_rp = cal_power_feature(signal, pick_rest_ch, freq_min=self.IAF_band[0],
                                                          freq_max=self.IAF_band[1], rp=True)
            rela_rest_power = (rest_power_rp - self.base_alpha_power) / self.base_alpha_power
            self.rela_rest_power_list.append(rela_rest_power)
            return rela_rest_power, rela_rest_power > self.rest_threshold
        else:  # Left / Right
            left_power = cal_power_feature(signal, self.left_ch, freq_min=8, freq_max=30)
            right_power = cal_power_feature(signal, self.right_ch, freq_min=8, freq_max=30)
            rela_left_power = (left_power - self.base_leftch_power) / self.base_leftch_power
            rela_right_power = (right_power - self.base_rightch_power) / self.base_rightch_power
            self.rela_left_power_list.append(rela_left_power)
            self.rela_right_power_list.append(rela_right_power)
            erd = rela_right_power - rela_left_power
            if self.label == StimType.Left:
                # erd = rela_left_power - rela_right_power
                return erd, erd < self.left_threshold
            else:
                # erd = rela_right_power - rela_left_power
                return erd, erd > self.right_threshold

    def up_threshold(self):
        if self.label == StimType.Rest:
            rela_rest_power = np.mean(self.rela_rest_power_list[-self.is_reached_buffer_len:-1])  # 正
            if self.rest_threshold < rela_rest_power - self.t_stride:
                # self.rest_threshold = rela_rest_power - self.t_stride
                self.rest_threshold = self.rest_threshold + self.t_stride
                self.rest_threshold_list.append([time.time() - self.t0, self.rest_threshold])
                print('Up threshold, Rest_threshold', self.rest_threshold)
        else:
            left_power = np.mean(self.rela_left_power_list[-self.is_reached_buffer_len:-1])
            right_power = np.mean(self.rela_right_power_list[-self.is_reached_buffer_len:-1])
            erd = right_power - left_power
            if self.label == StimType.Left:  # 左MI 右-左=负 下调
                if self.left_threshold > erd + self.t_stride:
                    self.left_threshold = erd + self.t_stride
                    self.left_threshold_list.append([time.time() - self.t0, self.left_threshold])
                    print('Up threshold. Left_threshold:', self.left_threshold)
            else:
                if self.right_threshold < erd - self.t_stride:  # 右MI 右-左=正
                    self.right_threshold = erd - self.t_stride
                    self.right_threshold_list.append([time.time() - self.t0, self.right_threshold])
                    print('Up threshold. Right_threshold:', self.right_threshold)

    def down_threshold(self):
        if self.label == StimType.Rest:
            rela_rest_power = np.mean(self.rela_rest_power_list[-self.is_reached_buffer_len:-1])
            if self.rest_threshold > (rela_rest_power + self.t_stride):
                # self.rest_threshold = rela_rest_power + self.t_stride
                self.rest_threshold = self.rest_threshold - self.t_stride
                self.rest_threshold_list.append([time.time() - self.t0, self.rest_threshold])
                print('Down threshold, Rest_threshold', self.rest_threshold)
        else:
            left_power = np.mean(self.rela_left_power_list[-self.is_reached_buffer_len:-1])
            right_power = np.mean(self.rela_right_power_list[-self.is_reached_buffer_len:-1])
            erd = right_power - left_power
            if self.label == StimType.Left:  # 左MI 右-左=负 上调
                if self.left_threshold < erd - self.t_stride:
                    self.left_threshold = erd - self.t_stride
                    self.left_threshold_list.append([time.time() - self.t0, self.left_threshold])
                    print('Down threshold, Left_threshold', self.left_threshold)
            else:
                if self.right_threshold > (erd + self.t_stride):
                    self.right_threshold = erd + self.t_stride
                    self.right_threshold_list.append([time.time() - self.t0, self.right_threshold])
                    print('Down threshold, Right_threshold', self.right_threshold)

    def get_result_log(self):
        score = sum(self.is_reached_buffer) / len(self.is_reached_buffer)
        score = round(score * 100, 2)
        self.trial_num = self.trial_num + 1
        print('trial:{:d}, cue:{}, score: {:.2f}%'.format(self.trial_num, self.label, score))
        self.result_log.append([self.trial_num, self.label, score])

    def save_log(self):
        stime = datetime.now().strftime('%H%M')
        if self.label == StimType.Rest and self.rest_threshold_list != []:
            rela_rest_pd = pd.DataFrame(data=self.rela_rest_power_list)
            rest_threshold_pd = pd.DataFrame(data=self.rest_threshold_list)
            rela_rest_pd.to_csv(self.save_path + r'/log/relative_rest_power_' + stime + '.csv', header=False, index=False)
            rest_threshold_pd.to_csv(self.save_path + r'/log/rest_threshold_' + stime + '.csv', header=False, index=False)
        elif self.right_threshold_list != [] or self.left_threshold_list != []:
            rela_right_pd = pd.DataFrame(data=self.rela_right_power_list)
            right_threshold_pd = pd.DataFrame(data=self.right_threshold_list)
            rela_right_pd.to_csv(self.save_path + r'/log/relative_right_power_' + stime + '.csv', header=False, index=False)
            right_threshold_pd.to_csv(self.save_path + r'/log/right_threshold_' + stime + '.csv', header=False, index=False)
            rela_left_pd = pd.DataFrame(data=self.rela_left_power_list)
            left_threshold_pd = pd.DataFrame(data=self.left_threshold_list)
            rela_left_pd.to_csv(self.save_path + r'/log/relative_left_power_' + stime + '.csv', header=False, index=False)
            left_threshold_pd.to_csv(self.save_path + r'/log/left_threshold_' + stime + '.csv', header=False, index=False)
        elif not self.result_log == []:
            trial_result_pd = pd.DataFrame(data=self.result_log)
            trial_result_pd.to_csv(self.save_path + r'/log/online_result_' + stime + '.csv', header=False, index=False)
        else:
            print('no log saved.')
        print('Online log saved successfully.')

def testmain():
    from MIdataset_NF import MIdataset
    from BCIConfig import ch_names, ch_types
    # dataset_path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean'
    # data_path = r'\S4\S4_20200721\NSsignal_2020_07_21_16_15_11.npz'
    p = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\wmm\wmm_20210126\Online_20210126_1525_33.npz'
    data = MIdataset(p)
    data.set_reference()
    signal = data.get_raw_data()
    signal1 = signal[10000:10500, :]
    p = Processor('1')
    p.label = 'Left'
    p.fs = 500
    p.left_ch = ['F3', 'F1', 'C2', 'C4', 'C6']
    p.right_ch = ['FC2', 'FC4', 'FC5', 'FC3', 'FC1']
    # p.ch_names = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1',
    #               'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'F5', 'AF3', 'AF4', 'P5',
    #               'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PO3', 'POz', 'PO4', 'Oz', 'F6']  # 41ch EEG
    # p.ch_types = ['eeg'] * 41
    p.info = mne.create_info(ch_names[:-2], p.fs, ch_types[:-2])
    p.info.set_montage('standard_1005')
    p.baseline_signal = signal[7000:9500, :]
    # p.base_power_alpha = p.cal_power_feature(p.baseline_signal, rest_ch, fmin=8, fmax=13)
    t0 = time.time()
    for i in range(100):
        time.sleep(1)
        p.base_power_ERDleft = cal_power_feature(p.baseline_signal, p.left_ch, freq_min=8, freq_max=30)
        p.base_power_ERDright = cal_power_feature(p.baseline_signal, p.right_ch, freq_min=8, freq_max=30)
        is_reached = p.is_reached_threshold(signal1)
        print(time.time()-t0, is_reached)
        t0 = time.time()
    # p.base_power_ERDright = p.cal_power_feature(p.baseline_signal, p.right_ch, fmin=8, fmax=30)
    # is_reached = p.is_reached_threshold(signal1)

if __name__ == '__main__':
    path = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\cax\cax_20210604\Baseline_post_ec.npz'
    data = dict(np.load(path, allow_pickle=True))
    baseline_eo = data['signal']
    print('1')