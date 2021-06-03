import numpy as np
from process_tools import csp_filter
from MIdataset_NF import MIdataset
# from MIdataset_nonNF import MIdataset


def iterative_CSP_LR(data_x, ch_names):
    # remove the least-discriminative channel
    # imput: unmixing CSP matrix (filter, channels)
    # return channels id, csp weight from small to large order
    csp = csp_filter(m=1)
    select_ch_num = 5
    ch_names = np.array(ch_names)
    # ch_name = ch_names
    # ch_z = [i for i, x in enumerate(ch_names) if x.endswith('z')]
    # ch_names = np.delete(ch_names, ch_z)
    # data_x = np.delete(data_x, ch_z, 1)
    for i in range(len(ch_names)):
        csp.fit(data_x, ['Left', 'Right'])
        abs_weight = abs(csp.csp_proj_matrix[0, :]) + abs(csp.csp_proj_matrix[-1, :])  # 每个通道first&last filter绝对值的和
        I = np.argsort(abs_weight)  # 排序
        left_ch_num = len([i for i in ch_names if isside(i, 'left')])
        right_ch_num = len([i for i in ch_names if isside(i, 'right')])
        if (isside(ch_names[I[0]], 'left') and left_ch_num > select_ch_num) or (
                isside(ch_names[I[0]], 'right') and right_ch_num > select_ch_num):
            ch_names = np.delete(ch_names, I[0])
            data_x = np.delete(data_x, I[0], 1)
        elif isside(ch_names[I[0]], 'left') and left_ch_num <= select_ch_num < right_ch_num:
            for i, ch in enumerate(ch_names[I]):
                if isside(ch, 'right'):
                    ch_names = np.delete(ch_names[I], i)
                    data_x = np.delete(data_x[:, I, :], i, 1)
                    break
        elif isside(ch_names[I[0]], 'right') and right_ch_num <= select_ch_num < left_ch_num:
            for i, ch in enumerate(ch_names[I]):
                if isside(ch, 'left'):
                    ch_names = np.delete(ch_names[I], i)
                    data_x = np.delete(data_x[:, I, :], i, 1)
                    break
        else:
            # ch_name_idx = (ch_name[:, None] == ch_names[I]).argmax(axis=0)
            leftlist, rightlist = [], []
            for i in range(len(I)):
                if isside(ch_names[I][i], 'left'):
                    leftlist.append(ch_names[I][i])
                else:
                    rightlist.append(ch_names[I][i])
            ch = np.array([leftlist, rightlist]).T
            # print(ch)
            return ch


def isside(ch_name, side):
    sideid = {'left': ('1', '3', '5', '7'), 'right': ('2', '4', '6', '8')}
    return ch_name.endswith(sideid[side])


def pipeline(p):
    # subject_set = os.listdir(dataset_path)
    subject_set = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
    df = pd.DataFrame(columns=['npz_file', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', ])
    for s_idx in range(len(subject_set)):
        # for s_idx in range(1):
        s_idx = 1
        s = subject_set[s_idx]
        sub_path = os.path.join(p, s)
        date_files = os.listdir(sub_path)
        k = 0
        for i in range(len(date_files)):
            date_file_path = os.path.join(sub_path, date_files[i])
            npz_files = os.listdir(date_file_path)
            # for j in range(len(npz_files)):
            for j in range(1):
                kappa_2class = [npz_files[j], ]
                data = MIdataset(os.path.join(date_file_path, npz_files[j]))
                data_x, label = data.get_epoch_data(select_label=['left', 'right'])
                ch = iterative_CSP_LR(data_x, label, ch_names=data.ch_names)
                kappa_2class.extend(ch.tolist())
                df.loc[k] = kappa_2class
                k = k + 1
        break
    df.to_excel('list_s2.xlsx')


if __name__ == '__main__':
    import os
    import pandas as pd
    # p = r'C:\StrokeEEGProj\codes\MIBCIProj_NF\data_set\S1\S1_20210120\Online_20210120_1341_07.npz'
    # p = r'D:\Myfiles\EEGProject\data_set\data_set_bcilab\healthy_subject\4class_large_add1\data_clean' \
    #     r'\S4\S4_20200722\NSsignal_2020_07_22_16_22_30.npz'
    # data = MIdataset(p)
    # data.bandpass_filter(1, 100)  # band pass
    # data.set_reference()  # CAR
    # data.removeEOGbyICA()  # ICA
    # data.bandpass_filter(8, 30)
    # select_ch = ['F3', 'F1', 'F2', 'F4', 'FC5', 'FC3', 'C5', 'C3', 'C1','CP4', 'CP6', 'F5',
    #              'P5', 'P3', 'P1', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CP2']
    # data_x, label = data.get_epoch_data(select_label=['Left', 'Right'], select_ch=select_ch)
    # ch = iterative_CSP_LR(data_x, label, ch_names=select_ch)
    # df = pd.DataFrame(data=ch)
    # df.to_csv(r'testcsv.csv', header=False, index=False)
    # df = pd.read_csv(r'D:\Myfiles\MIBCI_NF\process_tools\testcsv.csv', header=None)
    # left = list(df.iloc[:, 0].values)
    # print(left)
    # print(pd)
