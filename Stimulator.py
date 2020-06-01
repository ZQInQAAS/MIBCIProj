from threading import Thread, Event
from time import time
from BCIConfig import BCIEvent, StimType
from utils import PyPublisher


class Stimulator(PyPublisher, Thread):
    def __init__(self, stim_cfg):
        #  mne event (n_events, 3)
        #  first column: event time in samples
        #  third column: event id
        super(Stimulator, self).__init__()
        self.stim_config = stim_cfg
        self.class_list = list()
        self.stim_list = list()
        self.stim_sequence = list()
        self.__running = Event()  # 停止线程的标识
        self.__running.set()
        self.__flag = Event()  # 暂停线程的标识
        self.__flag.set()
        self.stim_sequence = self.stim_config.generate_stim_list()
        self.class_dict = self.stim_config.get_class_dict()

    def run(self):
        for i in range(len(self.stim_sequence)):
            if self.__running.isSet():
                stim, duration = self.stim_sequence[i]
                if stim in (s for s in StimType):
                    self.stim_list.append([time(), stim.name])
                    print(time(), stim.name)
                else:
                    self.class_list.append([time(), self.class_dict[stim]])
                    print(time(), stim)
                self.publish(BCIEvent.change_stim, stim)
                if stim == StimType.ExperimentStop:
                    self.publish(BCIEvent.stim_stop)
                self.__flag.clear()
                if duration != None:
                    self.__flag.wait(duration)
                    # sleep(duration)
                else:
                    self.__flag.wait()

    def get_gaze(self):
        self.__flag.set()

    def stop_stim(self):  # 界面中断
        self.__flag.set()
        self.__running.clear()
        self.stim_list.append((time(), StimType.Disconnect.value))
        self.publish(BCIEvent.stim_stop)
