from enum import Enum


class BCIEvent(Enum):
    readns_header = 1
    readns = 2
    change_stim = 3
    online_progressbar = 4
    online_ctrl = 5
    gaze_focus = 6
    cue_disconnect = 7
    stim_stop = 8


classStimList = ('left', 'right', 'rest')


class StimType(Enum):
    # left = 1
    # right = 2
    # rest = 3
    ExperimentStart = 10
    ExperimentStop = 11
    CrossOnScreen = 12
    StartOfTrial = 13
    EndOfTrial = 14
    EyeGaze = 15
    MoveUp = 16
    MoveDown = 17
    Still = 18
    Disconnect = 19
