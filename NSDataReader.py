import sys
import sched
import socket
import struct
import threading
import numpy as np
from time import sleep, time
from process_tools import RepeatingTimer


class NSDataReader(object):
    def __init__(self):
        self.address = '192.168.11.123', 9889  # 有线
        # self.address = '10.168.2.164', 9889  # QAAS_Bridge wifi
        # self.address = '192.168.43.166', 9889  # ZQ热点
        self.socket = socket.socket()
        self.socket.settimeout(15)
        self.repeat_timer = RepeatingTimer(0.01, self._read_data)  # 循环时间应<0.02
        self.signal = []
        self.data_time = []
        self.count = 0
        self.endTime = 0

    def start(self):
        self.socket.connect(self.address)
        self._read_header()
        self.repeat_timer.start()

    def _read_header(self):
        self._send_command_to_ns(3, 5)  # request for basic info
        sleep(0.1)
        # get basic information
        self.BasicInfo = self.socket.recv(1024)
        code = int.from_bytes(self.BasicInfo[4:6], byteorder='big')
        req = int.from_bytes(self.BasicInfo[6:8], byteorder='big')
        size = int.from_bytes(self.BasicInfo[8:12], byteorder='big')
        if code == 1 and req == 3 and size == 28 and len(self.BasicInfo) == 40:
            self.Bsize = int.from_bytes(self.BasicInfo[12:16], byteorder='little')
            self.BEegChannelNum = int.from_bytes(self.BasicInfo[16:20], byteorder='little')
            self.BEventChannelNum = int.from_bytes(self.BasicInfo[20:24], byteorder='little')
            self.BlockPnts = int.from_bytes(self.BasicInfo[24:28], byteorder='little')
            self.BSampleRate = int.from_bytes(self.BasicInfo[28:32], byteorder='little')
            self.BDataSize = int.from_bytes(self.BasicInfo[32:36], byteorder='little')
            self.BResolution = struct.unpack('<f', self.BasicInfo[36:40])[0]
            self.pattern = '<h' if self.BDataSize == 2 else '<i'
            self.ch_num = self.BEegChannelNum + self.BEventChannelNum
            # self.T = self.BlockPnts / self.BSampleRate / 2
        else:
            sys.exit(0)
        self._send_command_to_ns(2, 1)  # server/ start acquisition
        self._send_command_to_ns(3, 3)  # client/ request to start sending data

    def _read_data(self):
        while True:
            data_head = self.socket.recv(12)
            if len(data_head) != 0:
                break
        size = int.from_bytes(data_head[8:12], byteorder='big')  # 4400
        data = bytearray()
        while len(data) < size:
            data += self.socket.recv(size - len(data))
        data = [i[0] * self.BResolution for i in struct.iter_unpack(self.pattern, data)]  # (ch_num*20,)  1100
        self.signal += [data[i: i + self.ch_num - 1] for i in range(0, len(data), self.ch_num)]  # remove label column
        self.data_time.append(time())

    def stop_data_reader(self):
        self.repeat_timer.cancel()
        self._send_command_to_ns(3, 4)  # client/ request to stop sending data
        self._send_command_to_ns(2, 2)  # server/ stop acquisition
        self._send_command_to_ns(1, 2)  # general/ closing up connection
        sleep(1)
        self.socket.close()
        print('Close scan4.5 server successfully.')

    def get_ns_signal(self, duration=None):
        # signal: (sample, channal)
        if not self.signal:
            print('Get empty ns data.')
            return
        else:
            return self.signal[-duration:] if duration else self.signal

    def get_sample_rate(self):
        return self.BSampleRate

    def _send_command_to_ns(self, ctrcode, reqnum):
        a = 'CTRL'
        cmd = a.encode(encoding="utf-8")
        cmd += ctrcode.to_bytes(2, 'big')
        cmd += reqnum.to_bytes(2, 'big')
        cmd += (0).to_bytes(4, 'big')
        self.socket.sendall(cmd)


class NSDataReaderRandom(object):
    def __init__(self):
        self.ch_num = 26
        self.signal = []
        self.data_time = []
        self.fs = 500  # 数据生成速度无法达到500采样
        self.repeat_timer = RepeatingTimer(0.1, self._read_data)
        # path = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean' \
        #        r'\S4\S4_20200721\NSsignal_2020_07_21_16_05_04.npz'
        # npz_data_dict = dict(np.load(path))
        # data = npz_data_dict['signal']  # (samples, channels)
        # self.data = data[:, 0:27]
        # self.i = 0

    def start(self):
        self.repeat_timer.start()

    def _read_data(self):
        data = (10 * np.random.randn(520)).tolist()
        self.signal += [data[i: i + self.ch_num] for i in range(0, len(data), self.ch_num)]
        self.data_time.append(time())

    def stop_data_reader(self):
        print('Close neuroscan random server.')

    def get_ns_signal(self, duration=None):
        # self.signal = self.data[self.i*500:self.i*500+1000, :]
        # self.i = self.i+1
        # signal = np.array(self.signal)
        signal = self.signal
        return signal[-duration:] if duration else signal  # remove label column

    def get_sample_rate(self):
        return self.fs


if __name__ == '__main__':
    ns = NSDataReaderRandom()
    for i in range(10000):
        ns._read_data()
        s = ns.get_ns_signal(5)
        print(time())