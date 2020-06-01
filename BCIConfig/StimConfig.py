import random
from BCIConfig import classStimList, StimType

cue_path_set = [r'\cue_material\leftelbow_up_u3d.gif',
                r'\cue_material\rightelbow_up_u3d.gif',
                r'\cue_material\rest_img.png']
cue_path_set_return = [[r'\cue_material\left2.gif', r'\cue_material\left1.gif'],
                       [r'\cue_material\right2.gif', r'\cue_material\right1.gif'],
                       r'\cue_material\rest_img.png']


class StimConfig(object):
    def __init__(self):
        # self.withoutPredDur = 1  # without Predict Feedback Duration: 1s
        self.class_list = ['left', 'right', 'rest']
        self.each_class_num = 1
        self.baseline_duration = 2
        self.cue_interval_duration = 3
        self.display_cue_duration = 4
        self.is_repeat = False  # False：单向；True：双向重复
        self.set_cue_path()
        self.move_sound_path = r'../cue_material/move_sound.wav'
        self.relax_sound_path = r'../cue_material/relax_sound.wav'
        self.stop_sound_path = r'../cue_material/stop_sound.wav'

    def set_class(self, class_list):
        index = [classStimList.index(i) for i in classStimList if i in class_list]
        self.class_list = [classStimList[i] for i in index]
        self.set_cue_path()
        self.cue_path = [self.cue_path[i] for i in index]

    def set_cue_path(self):
        self.cue_path = cue_path_set if not self.is_repeat else cue_path_set_return

    def generate_stim_list(self):
        stim_list = self.shuffle_stim(self.class_list * self.each_class_num)
        stim_sequence = list()
        stim_sequence.append((StimType.ExperimentStart, self.baseline_duration))
        for i in range(len(stim_list)):
            # stim_sequence.append((StimType.EyeGaze, None))
            stim_sequence.append((StimType.StartOfTrial, 0))
            stim_sequence.append((stim_list[i], 0))
            if stim_list[i] == 'rest':
                stim_sequence.append((StimType.Still, self.display_cue_duration))
            else:
                if self.is_repeat:
                    stim_sequence.append((StimType.MoveUp, self.display_cue_duration / 2))
                    stim_sequence.append((StimType.MoveDown, self.display_cue_duration / 2))
                else:
                    stim_sequence.append((StimType.MoveUp, self.display_cue_duration))
            # stim_sequence.append((StimType.EndOfTrial, self.cue_interval_duration + random.uniform(-1, 1)))  # 随机间隔
            stim_sequence.append((StimType.EndOfTrial, self.cue_interval_duration))
        stim_sequence.append((StimType.ExperimentStop, 0))
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
                'display_cue_duration': self.display_cue_duration,
                'is_repeat': self.is_repeat}


if __name__ == '__main__':
    stim = StimConfig()
    seq = stim.generate_stim_list()
    print(stim.class_list)
