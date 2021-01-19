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
