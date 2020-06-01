import os
import datetime


com_num_set = {'elbow': {'left': 53, 'right': 53},
               'wrist': {'left': 1, 'right': 1},
               'forearm': {'left': 1, 'right': 1},
               'hand': {'left': 1, 'right': 1}}
# 最高/左  最低/右
exo_position = {'elbow': {'left': {'highest_point': 1250, 'lowest_point': 250},
                          'right': {'highest_point': 1400, 'lowest_point': 2200}},
                'wrist': {'left': {'highest_point': 50, 'lowest_point': 50},
                          'right': {'highest_point': 50, 'lowest_point': 50}},
                'forearm': {'left': {'highest_point': 50, 'lowest_point': 50},
                            'right': {'highest_point': 50, 'lowest_point': 50}},
                'hand': {'left': {'highest_point': 50, 'lowest_point': 50},
                         'right': {'highest_point': 50, 'lowest_point': 50}}}
exo_velocity = {'elbow': {'left': 30, 'right': 30},
                'wrist': {'left': 5, 'right': 5},
                'forearm': {'left': 5, 'right': 5},
                'hand': {'left': 5, 'right': 5}}


class SubjectInfoConfig(object):
    def __init__(self, subject_name=None):
        self.subject_name = subject_name
        self.affected_side = 'left'
        self.exo_type = 'elbow'
        self.parent_dir = os.path.abspath(os.getcwd())
        self.dataset_dir = self.parent_dir + r'/data_set'
        self.make_dir(self.dataset_dir)
        # 最高/左  最低/右
        self.exo_position = {'elbow': {'left': {'highest_point': None, 'lowest_point': None},
                                       'right': {'highest_point': None, 'lowest_point': None}},
                             'wrist': {'left': {'highest_point': None, 'lowest_point': None},
                                       'right': {'highest_point': None, 'lowest_point': None}},
                             'forearm': {'left': {'highest_point': None, 'lowest_point': None},
                                         'right': {'highest_point': None, 'lowest_point': None}},
                             'hand': {'left': {'highest_point': None, 'lowest_point': None},
                                      'right': {'highest_point': None, 'lowest_point': None}}}

        self.exo_velocity = {'elbow': {'left': None, 'right': None},
                             'wrist': {'left': None, 'right': None},
                             'forearm': {'left': None, 'right': None},
                             'hand': {'left': None, 'right': None}}

    def set_subject(self, subject_name):
        self.subject_name = subject_name
        self.subject_dir = self.dataset_dir + '/' + self.subject_name
        if os.path.exists(self.dataset_dir):
            self.make_dir(self.subject_dir)
            # self.set_date_dir()

    def set_date_dir(self):
        date = datetime.datetime.now().strftime('%Y%m%d')
        if os.path.exists(self.subject_dir):
            self.date_dir = self.subject_dir + '/' + self.subject_name + '_' + date
            self.make_dir(self.date_dir)

    def get_date_dir(self):
        return self.date_dir

    def get_dataset_path(self):
        return self.dataset_dir

    def get_model_path(self, subject_name=None):
        return self.subject_dir + '/' + subject_name + '_model.pkl' \
            if subject_name else self.subject_dir + '/' + self.subject_name + '_model.pkl'

    def get_param_path(self, subject_name=None):
        return self.subject_dir + '/' + subject_name + '_param.pkl' \
            if subject_name else self.subject_dir + '/' + self.subject_name + '_param.pkl'

    def get_com_num(self):
        return com_num_set[self.exo_type][self.affected_side]

    def set_exo_position(self, high, low):
        self.exo_position[self.exo_type][self.affected_side]['highest_point'] = high
        self.exo_position[self.exo_type][self.affected_side]['lowest_point'] = low

    def get_exo_position(self):
        high = self.exo_position[self.exo_type][self.affected_side]['highest_point']
        high = high if high else exo_position[self.exo_type][self.affected_side]['highest_point']
        low = self.exo_position[self.exo_type][self.affected_side]['lowest_point']
        low = low if low else exo_position[self.exo_type][self.affected_side]['lowest_point']
        return high, low

    def set_exo_velocity(self, v):
        self.exo_velocity[self.exo_type][self.affected_side] = v

    def get_exo_velocity(self):
        v = self.exo_velocity[self.exo_type][self.affected_side]
        return v if v else exo_velocity[self.exo_type][self.affected_side]

    def make_dir(self, path):
        if not os.path.exists(path):  # 创建 姓名_日期 文件夹
            os.makedirs(path)
