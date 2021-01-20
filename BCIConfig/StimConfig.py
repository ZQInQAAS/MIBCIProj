import random
from BCIConfig import StimType

# cue_paths = {'cross_blue': r'\cue_material\cross_blue.png',
#              'cross_white': r'\cue_material\cross_white.png',
#              'left': r'\cue_material\left_hand.png',
#              'right': r'\cue_material\right_hand.png',
#              'smiley01': r'\cue_material\smiley01.png',
#              'smiley02': r'\cue_material\smiley02.png',
#              'smiley03': r'\cue_material\smiley03.png'}


class StimConfig(object):
    def __init__(self):
        self.class_list = ['Left', 'Right', 'Rest']
        self.each_class_num = 2
        self.baseline_duration = 4  # 1 min
        self.cue_interval_duration = 3
        self.display_cue_duration = 5
        self.NF_training_duration = 305  # 5 min
        self.move_sound_path = r'../cue_material/move_sound.wav'
        self.relax_sound_path = r'../cue_material/relax_sound.wav'
        self.stop_sound_path = r'../cue_material/stop_sound.wav'

    def generate_stim_list(self, session_type):
        stim_sequence = list()
        stim_sequence.append((StimType.ExperimentStart, 2))
        stim_sequence.append((StimType.CrossOnScreen, self.baseline_duration))
        stim_sequence.append((StimType.EndOfBaseline, 2))
        if session_type in ['Acq', 'Online']:
            stim_list = self.shuffle_stim(self.class_list * self.each_class_num)
            for i in range(len(stim_list)):
                stim_sequence.append((StimType.CrossOnScreen, 1))
                if session_type == 'Acq':
                    stim_sequence.append((StimType[stim_list[i]], self.display_cue_duration))
                else:
                    stim_sequence.append((StimType[stim_list[i]], 1))
                    if stim_list[i] == 'Rest':
                        stim_sequence.append((StimType.RestNF, self.display_cue_duration - 1))
                    else:
                        stim_sequence.append((StimType.LRNF, self.display_cue_duration - 1))
                stim_sequence.append((StimType.EndOfTrial, self.cue_interval_duration + random.randint(0, 1)))  # 随机间隔
        elif session_type == 'Rest_nonNF':
            stim_sequence.append((StimType.Rest, self.NF_training_duration))
        elif session_type == 'LR_nonNF':
            stim_sequence.append((StimType.LRCue, self.NF_training_duration))
        elif session_type == 'RestNF':
            stim_sequence.append((StimType.Rest, 5))
            stim_sequence.append((StimType.RestNF, self.NF_training_duration - 5))
        else:  # LRNF
            stim_sequence.append((StimType.LRCue, 5))
            stim_sequence.append((StimType.LRNF, self.NF_training_duration - 5))
        stim_sequence.append((StimType.ExperimentStop, 1))
        return stim_sequence

    def shuffle_stim(self, stim_list):
        times = 2  # 几个为一组进行随机
        for i in range(int(self.each_class_num / times)):
            random_width = times * len(self.class_list)
            s_list = stim_list[i * random_width:(i + 1) * random_width]
            random.shuffle(s_list)
            stim_list[i * random_width:(i + 1) * random_width] = s_list
        return stim_list

    def get_class_dict(self):
        return {x: i for i, x in enumerate(self.class_list)}

    def get_stim_pram(self):
        return {'class_list': self.class_list,
                'each_class_num': self.each_class_num,
                'baseline_duration': self.baseline_duration,
                'cue_interval_duration': self.cue_interval_duration,
                'display_cue_duration': self.display_cue_duration}


if __name__ == '__main__':
    import numpy as np
    stim = StimConfig()
    seq = stim.generate_stim_list('online')
    s = np.array(seq)
    print(seq)
