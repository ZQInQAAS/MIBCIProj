from threading import Thread, Event
from time import sleep, time
from BCIEnum import BCIEvent, StimType


class Stimulator(object):
    def __init__(self, pybus, stim_cfg):
        #  mne event (n_events, 3)
        #  first column: event time in samples
        #  third column: event id
        self.stim_config = stim_cfg
        self.pybus = pybus
        self.class_list = list()
        self.stim_list = list()
        self.stim_sequence = list()
        self.thread_stim = Thread(target=self.stim_run)
        self.wait_event = Event()
        self.wait_event.set()

    def start(self):
        self.stim_sequence = self.stim_config.generate_stim_list()
        self.class_dict = self.stim_config.get_class_dict()
        self.thread_stim.start()

    def stim_run(self):
        for i in range(len(self.stim_sequence)):
            stim, duration = self.stim_sequence[i]
            if stim in StimType:
                self.stim_list.append([time(), stim.name])
                # print(time(), stim.name)
            else:
                self.class_list.append([time(), self.class_dict[stim]])
                # print(time(), stim)
            self.pybus.publish(self, BCIEvent.change_stim, stim)
            if stim == StimType.ExperimentStop:
                self.pybus.publish(self, BCIEvent.save_data)
            self.wait_event.clear()
            if duration is not None:
                self.wait_event.wait(duration)
                # sleep(duration)
            else:
                self.wait_event.wait()

    def get_gaze(self):
        self.wait_event.set()

    def stop_stim(self):  # 界面中断
        self.stim_list.append((time(), StimType.Disconnect.value))
        self.pybus.publish(self, BCIEvent.save_data)
        # self.thread_stim.join()
