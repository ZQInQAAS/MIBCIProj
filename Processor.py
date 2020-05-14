import pickle
from utils import PyPublisher
from time import time, strftime
from BCIEnum import BCIEvent, StimType
from NSDataReader import RepeatingTimer


class Processor(PyPublisher):
    def __init__(self, main_cfg):
        PyPublisher.__init__(self)
        self.class_list = main_cfg.stim_cfg.class_list
        self.model_path = main_cfg.subject.get_model_path()
        self.save_path = main_cfg.subject.get_date_dir()
        self.epoch_dur = 0.2  # 每多少秒判断一次
        self.proc_bar_len = main_cfg.stim_cfg.display_cue_duration/self.epoch_dur
        self.online_timer = RepeatingTimer(self.epoch_dur, self.online_run)
        self.signal = None
        self.predict_state = False
        self.is_rest = False
        self.trial_num = 0
        self.right_num_one_run = []
        self.right_num_all = []
        self.result_log = []

    def start(self):
        with open(self.model_path, 'rb') as f:
            self.clf = pickle.load(f)
        self.online_timer.start()
        ns_header = self.publish(BCIEvent.readns_header)
        self.fs = ns_header['sample_rate']

    def handle_stim(self, stim):
        if stim in self.class_list:
            self.label = stim
            self.label_y = self.class_list.index(stim)
            self.is_rest = True if stim == 'rest' else False
            return
        if stim == StimType.StartOfTrial:
            self.predict_state = True
            return
        if stim == StimType.EndOfTrial:
            self.predict_state = False
            self.get_result_log()
            return

    def online_run(self):
        if self.predict_state:
            signal = self.publish(BCIEvent.readns, duration=500)
            result = self.clf.online_predict(signal, self.fs)
            predict_right = 1 if result == self.label_y else 0
            self.right_num_one_run.append(predict_right)
            if predict_right:
                score = sum(self.right_num_one_run) / self.proc_bar_len
                self.publish(BCIEvent.online_progressbar, score)
            if not self.is_rest and (len(self.right_num_one_run) == 1 or self.right_num_one_run[-2] != predict_right):
                self.publish(BCIEvent.online_ctrl, bool(predict_right))

    def get_result_log(self):
        epoch_num = len(self.right_num_one_run)
        one_run_acc = sum(self.right_num_one_run) / epoch_num
        self.right_num_all.extend(self.right_num_one_run)
        self.right_num_one_run = []
        all_acc = one_run_acc if len(self.right_num_all) == 0 else sum(self.right_num_all) / len(self.right_num_all)
        self.trial_num = self.trial_num + 1
        # print('————————————')
        one_run_log = '\ntrial ' + str(self.trial_num) + ':  ' + 'cue:' + str(self.label) + \
                      '\nthis run acc:' + str(one_run_acc) + ', epoch num:' + str(epoch_num) + \
                      '\nall runs acc:' + str(all_acc)
        self.result_log.append(one_run_log)
        # print(one_run_log)

    def save_log(self):
        path = strftime(self.save_path + "//onlineResult_%Y%m%d_%H%M_%S.txt")
        with open(path, 'a') as f:
            f.writelines(self.result_log)
        print('Online log saved successfully.')
