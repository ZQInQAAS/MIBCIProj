from enum import Enum


class BCIEvent(Enum):
    readns_header = 1
    readns = 2
    change_stim = 3
    online_bar = 4
    online_face = 5
    gaze_focus = 6
    cue_disconnect = 7
    stim_stop = 8
    MRsubmit = 9


class StimType(Enum):
    Left = 1
    Right = 2
    Rest = 3
    LRCue = 4
    RestNF = 5
    LRNF = 6
    StartOfMR = 7
    AnswerOfMR = 8  # 开始答题
    Statement = 9
    ExperimentStart = 10
    ExperimentStop = 11
    CrossOnScreen = 12
    StartOfTrial = 13
    EndOfTrial = 14
    EndOfBaseline = 15
    # MoveUp = 16
    # MoveDown = 17
    # Still = 18
    Disconnect = 19


events_id_3mi = {'Left': 1, 'Right': 2, 'Rest': 3}
event_id_mrt = {'StartOfMR': 7}
fs = 500

# ch_types = ['eeg'] * 52 + ['eog'] * 2

# ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
#             'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
#             'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6',
#             'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2', 'M1', 'M2', 'VEOG', 'HEOG']  # 50eeg+2ref+2eog
ch_types = ['eeg'] * 62 + ['eog'] * 2
ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'M2',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'O1', 'Oz', 'O2', 'HEOG', 'VEOG']  # 60eeg+2ref+2eog(H水平 V竖直)

ch_types60 = ['eeg'] * 60
ch_names60 = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
              'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
              'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
              'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
              'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
              'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
              'O1', 'Oz', 'O2']  # 60eeg
ch_types62 = ['eeg'] * 60 + ['eog'] * 2
ch_names62 = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
              'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
              'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
              'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
              'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
              'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
              'O1', 'Oz', 'O2', 'HEOG', 'VEOG']  # 60eeg+2eog(H水平 V竖直)

montage1020 = ['Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
               'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10',
               'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
               'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
               'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
               'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
               'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
               'O1', 'Oz', 'O2', 'O9', 'Iz', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2']  # 94ch

pick_rest_ch = ['AF3', 'AF4',  # 'Fp1', 'Fpz', 'Fp2',
                'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
                'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6']  # 19ch

pick_motor_ch = ['F5', 'F3', 'F1', 'F2', 'F4', 'F6',
                 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
                 'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
                 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6',
                 'P5', 'P3', 'P1', 'P2', 'P4', 'P6']

#  NS random
# ch_names = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1',
#             'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']  # NS random
# ch_types = ['eeg'] * 26
# pick_rest_ch = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz']  # NS random
# pick_motor_ch = ['F3', 'F1', 'F2', 'F4', 'FC3', 'FC1', 'FC2', 'FC4',
#                  'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6']

# MRcorrAns1 = [(1, 3), (1, 4), (2, 4), (2, 3), (1, 3),
#               (1, 4), (2, 4), (2, 3), (3, 4), (1, 4)]
# MRcorrAns2 = [(2, 4), (2, 4), (2, 4), (1, 4), (2, 4),
#               (2, 3), (1, 3), (1, 4), (2, 4), (2, 3)]

MRcorrAns1 = [(1, 3), (1, 4), (2, 4), (2, 3), (2, 3),
              (2, 4), (2, 4), (2, 4), (1, 3), (2, 4)]  # modified version by Moè, A. (2021)
MRcorrAns2 = [(1, 3), (1, 4), (2, 4), (1, 4), (2, 4),
              (1, 4), (2, 3), (2, 3), (1, 4), (2, 3)]

subject_set22 = ['PNM', 'XY', 'CYH', 'WRQ', 'ZXY', 'YZT', 'WXH', 'LXT', 'FX', 'SXF', 'WCC',
                 'HYD', 'XW', 'WYQ', 'CQY', 'LY', 'MYH', 'MHJ', 'LYR', 'WY', 'CYJ', 'CZ']  # 22 subject

subject_paf22 = [9.5, 10, 10, 10, 11, 10.25, 10.5, 10.5, 10.25, 10.25, 9.75,
                 9.75, 9, 9.5, 10.5, 10, 9.75, 9.25, 9.5, 10.5, 10, 10.5]
