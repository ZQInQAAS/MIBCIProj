import numpy as np
import pandas as pd
from time import time, sleep
from threading import Thread, Event
from process_tools import PyPublisher
from BCIConfig import BCIEvent, StimType


class Stimulator(PyPublisher, Thread):
    def __init__(self, main_cfg):
        super(Stimulator, self).__init__()
        self.stim_config = main_cfg.stim_cfg
        self.stim_sequence = self.stim_config.generate_stim_list(main_cfg.session_type)
        self.class_list = list()
        self.stim_list = list()
        self.__running = Event()  # 停止线程的标识
        self.__running.set()
        self.__flag = Event()  # 暂停线程的标识
        self.__flag.set()

    def run(self):
        for i in range(len(self.stim_sequence)):
            if self.__running.isSet():
                stim, duration = self.stim_sequence[i]
                self.stim_list.append([time(), stim.name])
                print(time(), stim.name)
                if stim.name in self.stim_config.class_list:
                    self.class_list.append([time(), stim.value])
                self.publish(BCIEvent.change_stim, stim)
                if stim == StimType.ExperimentStop:
                    self.publish(BCIEvent.stim_stop)
                    sleep(3)
                self.__flag.clear()
                self.__flag.wait(duration)

    def get_stimdata(self, data_time, samples_num):
        # return mne event(n_events, 3) First column: event time in sample; Third column: event id
        first_time, last_time = data_time[0], data_time[-1]
        data_time = np.linspace(0, last_time - first_time, samples_num)
        if self.class_list != []:
            class_list = np.array(self.class_list)
            class_list[:, 0] = class_list[:, 0] - first_time  # error: too many indices for array
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
        else:
            events = None
        stim_log = pd.DataFrame(self.stim_list)
        stim_log.iloc[:, 0] = stim_log.iloc[:, 0] - first_time
        return events, stim_log

    def stop_stim(self):  # 界面中断
        self.__flag.set()
        self.__running.clear()
        self.stim_list.append((time(), StimType.Disconnect.value))
        self.publish(BCIEvent.stim_stop)
