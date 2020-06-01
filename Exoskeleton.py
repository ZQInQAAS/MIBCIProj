# -*- coding: utf-8 -*-
import time
import serial
import threading
from BCIConfig import StimType


# 香港理工机械手控制
class Exoskeleton(object):
    def __init__(self, subject):
        self.goal = 1
        self.last = 0
        self.count = 0
        self.command_num = 0
        self.last_command = 'stop'
        self.command_num = 0
        self.baud_rate = 9600
        self.Connected = False
        self.subject = subject
        self.is_exo_feedback = False
        self.com_write_lock = threading.Lock()
        self.is_online = False
        self.first_stage = True
        # self.set_exo_param()

    def set_exo_param(self):
        self.exo_type = self.subject.exo_type
        self.affectedSide = self.subject.affected_side
        highest_point, lowest_point = self.subject.get_exo_position()
        velocity = self.subject.get_exo_velocity()
        self.set_param(highest_point, lowest_point, velocity)

    def set_param(self, highest_point, lowest_point, velocity):
        self.highest_point = highest_point
        self.lowest_point = lowest_point
        self.velocity = velocity
        if self.exo_type in ['forearm', 'wrist']:
            self.velocity_str = '%02d' % self.velocity
            self.highest_point_str = '%02d' % self.highest_point
            self.lowest_point_str = '%02d' % self.lowest_point

    def connect(self, com_num):
        self.set_exo_param()
        self.com_num = com_num
        if not self.Connected:
            self.link_to_exoskeleton()
            self.Connected = True
        self.reset_arm()

    def handle_stim(self, stim):
        if not self.is_exo_feedback:
            return
        if stim == StimType.ExperimentStart:
            self.connect(self.subject.get_com_num())
        if stim == StimType.MoveUp:
            self.first_stage = True if self.is_online else self.stretch_arm()
            return
        if stim == StimType.MoveDown:
            self.first_stage = False if self.is_online else self.back_arm()
            return
        if stim == StimType.EndOfTrial:
            self.reset_arm()
            return
        if stim == StimType.ExperimentStop:
            self.disconnect_com()

    def online_feedback(self, predict):
        if not self.is_exo_feedback:
            return
        if predict:
            self.back_arm() if self.first_stage else self.stretch_arm()
        else:
            self.exoskeleton_stop()

    # 连接串口，输入串口号和波特率
    def link_to_exoskeleton(self):
        print('Trying to Connect to com')
        try:
            self.com = serial.Serial('com' + str(self.com_num), self.baud_rate)
            print('Connect to com successfully')
        except Exception as e:
            print('Open serial failed.' + str(e))

    # 读取当前位置
    def read_position(self):
        self.com_write("APE")
        time.sleep(0.01)
        position = int(self.com.read(4).decode())
        self.com.read_all()
        return position

    def strategy_reciprocating(self):
        now_position = self.read_position()
        if abs(self.last - now_position) == 0:
            if self.goal == 1:
                self.back_arm()
                self.goal = 2
            else:
                self.stretch_arm()
                self.goal = 1
        self.last = now_position

    # 预览 循环运动
    def preview_step(self):
        try:
            if self.exo_type == 'elbow':
                i = 0
                while i < 30:
                    self.strategy_reciprocating()
                    time.sleep(0.2)
                    i += 1
                # self.back_arm()
                self.stretch_arm()
            else:
                time.sleep(1)
                reciprocate_command = ''
                if self.exo_type == 'forearm':
                    reciprocate_command = 'A3V' + self.velocity_str + 'L' + self.lowest_point_str + 'R' + self.highest_point_str + 'E'
                elif self.exo_type == 'wrist':
                    reciprocate_command = 'A4V' + self.velocity_str + 'L' + self.lowest_point_str + 'R' + self.highest_point_str + 'E'
                self.com_write(reciprocate_command)
                print(reciprocate_command)
                time.sleep(5)
        except:
            print("Exoskeleton not connected.")

    # 停止运动
    def exoskeleton_stop(self):
        stop_command = ''
        if self.exo_type == 'elbow':
            if self.affectedSide == 'left':
                self.is_left = True
            else:
                self.is_left = False
            position_diff = -20
            position = self.read_position()
            stop_command = "AS" + str(self.velocity) + "G" + str(position + position_diff) + "E"
            # self.last_command = 'stop'
        elif self.exo_type in ['forearm', 'wrist']:
            stop_command = 'A0V00L00R00E'
        self.com_write(stop_command)
        # print("stop")

    # 展臂动作
    def stretch_arm(self):
        stretch_command = ''
        if self.exo_type == 'elbow':
            stretch_command = "AS" + str(self.velocity) + "G" + str(self.lowest_point) + "E"
        elif self.exo_type in['forearm', 'wrist']:
            if self.exo_type == 'forearm':
                stretch_command = 'A1V' + self.velocity_str + 'L' + self.lowest_point_str + 'R00E'
            else:
                stretch_command = 'A2V' + self.velocity_str + 'L' + self.lowest_point_str + 'R00E'
        self.com_write(stretch_command)
        print("stretch ", stretch_command)

    # 收臂动作
    def back_arm(self):
        back_command = ''
        if self.exo_type == 'elbow':
            back_command = "AS" + str(self.velocity) + "G" + str(self.highest_point) + "E"
        elif self.exo_type in ['forearm', 'wrist']:
            if self.exo_type == 'forearm':
                back_command = 'A1V' + self.velocity_str + 'L00R' + self.highest_point_str + 'E'
            else:
                back_command = 'A2V' + self.velocity_str + 'L00R' + self.highest_point_str + 'E'
        self.com_write(back_command)
        print("back ", back_command)

    # 快速复位动作
    def reset_arm(self):
        reset_command = ''
        if self.exo_type == 'elbow':
            reset_command = "AS" + str(self.velocity + 10) + "G" + str(self.lowest_point) + "E"
        elif self.exo_type in ['forearm', 'wrist']:
            if self.exo_type == 'forearm':
                reset_command = 'A1V' + self.velocity_str + 'L00R01E'
            else:
                reset_command = 'A2V' + self.velocity_str + 'L00R01E'
        self.com_write(reset_command)
        print("reset ", reset_command)

    # 关闭串口
    def disconnect_com(self):
        self.reset_arm()
        self.com.close()
        self.Connected = False
        print("Exoskeleton close successfully")

    def com_write(self, message):
        self.com_write_lock.acquire()
        self.com.write(message.encode())
        self.com_write_lock.release()

