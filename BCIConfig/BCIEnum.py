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


class StimType(Enum):
    Left = 1
    Right = 2
    Rest = 3
    LRCue = 4
    RestNF = 5
    LRNF = 6
    ExperimentStart = 10
    ExperimentStop = 11
    CrossOnScreen = 12
    StartOfTrial = 13
    EndOfTrial = 14
    EndOfBaseline = 15
    MoveUp = 16
    MoveDown = 17
    Still = 18
    Disconnect = 19


event_id = {'Left': 1,
            'Right': 2,
            'Rest': 3}

ch_types = ['eeg'] * 52 + ['eog'] * 2

ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6',
            'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2', 'M1', 'M2', 'VEOG', 'HEOG']  # 50eeg+2ref+2eog

pick_rest_ch = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4',
                'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
                'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']

pick_motor_ch = ['F5', 'F3', 'F1', 'F2', 'F4', 'F6',
                 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
                 'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
                 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6',
                 'P5', 'P3', 'P1', 'P2', 'P4', 'P6']
