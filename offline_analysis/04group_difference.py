import numpy as np
import pandas as pd
from process_tools import plot_topo, plot_group_boxplot, plot_subject_barplot, pvalue_1D, pvalue_2D


def drop_sub(data_df, sub_list, col_name='Subject_name'):
    # 移除部分被试
    data_df_sub = data_df.drop(data_df[data_df[col_name].isin(sub_list)].index)
    data_df_sub.reset_index(drop=True, inplace=True)
    return data_df_sub


def load_data(ab_path, data_path):
    acc_csv = pd.read_csv(ab_path + data_path)
    bad_sub = ['LYR', 'CYH', 'LXT', 'FX']
    acc_csv_sub = drop_sub(acc_csv, bad_sub, 'Subject_name')
    return acc_csv_sub


def group_pipeline():
    # 组间差异及显著性 箱线图
    acc_path = r'\Acc_raw\df_class123_tenfold_Acc_60ch_raw.csv'
    acc_csv = load_data(ab_path, acc_path)
    acc_csv['Subject_id'] = range(1, len(acc_csv) + 1)
    plot_group_boxplot(acc_csv.loc[:, ['Pre', 'Post']])
    groupA, groupB = acc_csv['Pre'], acc_csv['Post']
    p = pvalue_1D(groupA, groupB, ispaired=True)
    print('group p value:', p)


def draw_acc():
    # 并列柱状图 被试在NFT前后 acc表现变化
    acc_path = r'\Acc_raw\df_class123_tenfold_Acc_60ch_raw.csv'
    acc_csv = load_data(ab_path, acc_path)
    acc_csv.index = range(1, len(acc_csv) + 1)
    acc_csv_sub = acc_csv.loc[:, ['Pre', 'Post']]
    acc_csv_stack = acc_csv_sub.stack()
    acc_csv_stack = acc_csv_stack.reset_index()  # index变成数据列
    acc_csv_stack.columns = ['Subject_id', 'Stage', 'Accuracy']
    plot_subject_barplot(acc_csv_stack, x='Subject_id', y='Accuracy', hue='Stage')


def baseline_power():
    # NFT前后Baseline rela power变化
    rhythm = 'gamma'
    rpower_path = r'\Baseline_power\df_rela_power_Bas_' + rhythm + '_60s_1.csv'
    rp_csv = load_data(ab_path, rpower_path)
    pre_eo = rp_csv[rp_csv['Run'] == 'pre_eo']
    post_eo = rp_csv[rp_csv['Run'] == 'post_eo']
    vmin, vmax, cmin, cmax, rmin, rmax = _load_minmax(rhythm)
    bad_ch = ['T7', 'T8', 'TP7', 'TP8']
    pre = pre_eo.mean().drop('Subject_id', axis=0)
    post = post_eo.mean().drop('Subject_id', axis=0)
    change = post - pre
    rela_change = change / pre
    plot_topo(pre, vmin, vmax, drop_ch=bad_ch, figname=rhythm+'_pre')
    plot_topo(post, vmin, vmax, drop_ch=bad_ch, figname=rhythm+'_post', showcolorbar=True)
    plot_topo(change, cmin, cmax, drop_ch=bad_ch, cmap='RdBu_r', figname=rhythm+'_change', showcolorbar=True)
    plot_topo(rela_change, rmin, rmax, drop_ch=bad_ch, cmap='RdBu_r', figname='a_'+rhythm+'_relachange', showcolorbar=True)


def neuroFB_power():
    # NFT期间 rela power变化
    bas_rpower_path = r'\Baseline_power\df_rela_power_Bas_alpha_60s_1.csv'
    bas_rp_csv = load_data(ab_path, bas_rpower_path)
    pre_eo = bas_rp_csv[bas_rp_csv['Run'] == 'pre_eo']
    prebase_rp = pre_eo.mean().drop('Subject_id', axis=0)
    rpower_path = r'\RestNFT_power\df_rela_power_Res_alpha_60s_1.csv'
    bad_ch = ['T7', 'T8', 'TP7', 'TP8']
    rp_csv = load_data(ab_path, rpower_path)
    vmin, vmax = 0.036, 0.273  # relative power
    cmin, cmax = -0.004, 0.115  # relative power change
    rmin, rmax = -0.029, 0.719  # relative power relative change
    for i in range(1, 7):  # 6 run
        onerun = rp_csv[rp_csv['Run'] == i]
        onerunrp = onerun.mean().drop(['Subject_id', 'Run'], axis=0)
        change = onerunrp - prebase_rp
        rela_change = change / prebase_rp
        # _print_minmax(change)
        plot_topo(onerunrp, vmin, vmax, drop_ch=bad_ch, figname='alphaRP_run' + str(i), showcolorbar=True)
        plot_topo(change, cmin, cmax, drop_ch=bad_ch, figname='alphaRPchange_run' + str(i), showcolorbar=True)
        plot_topo(rela_change, rmin, rmax, drop_ch=bad_ch, figname='alphaRPrelachange_run' + str(i), showcolorbar=True)

def Acq_power():
    label = 'Left'
    acqpre_rpower_path = r'\Acq_power\df_rela_power_Acq_pre' + label + '_alpha.csv'
    acqpre_rp_csv = load_data(ab_path, acqpre_rpower_path)
    acq_rpower_path = r'\Acq_power\df_rela_power_Acq_' + label + '_alpha.csv'
    acq_rp_csv = load_data(ab_path, acq_rpower_path)
    bad_ch = ['T7', 'T8', 'TP7', 'TP8']
    # plot_power(acqpre_rp_csv, acq_rp_csv, Acq_run_list, label, bad_ch)
    # plot_power_change(acqpre_rp_csv, acq_rp_csv, bad_ch, label)
    plot_power_p(acqpre_rp_csv, acq_rp_csv, bad_ch, label)

def plot_power(acqpre_rp_csv, acq_rp_csv, label, bad_ch):
    Acq_run_list = ['post1', 'post2', 'post3', 'pre1', 'pre2', 'pre3']
    vmin, vmax, cmin, cmax, rmin, rmax = _load_minmax_bylabel(label)
    for i in range(6):  # 6 run
        # i = 2  # post3
        onerunrppre, onerunrp, change, rela_change = _cal_relapower(acqpre_rp_csv, acq_rp_csv, label=Acq_run_list[i])
        # _print_minmax(onerunrppre, bad_ch)
        # _print_minmax(onerunrp, bad_ch)
        figname = [label + '_' + Acq_run_list[i], 'pre' + label + '_' + Acq_run_list[i],
                   label + '_change_' + Acq_run_list[i], label + '_relachange_' + Acq_run_list[i]]
        plot_topo(onerunrp, vmin, vmax, drop_ch=bad_ch, figname=figname[0], showcolorbar=True)
        plot_topo(onerunrppre, vmin, vmax, drop_ch=bad_ch, figname=figname[1], showcolorbar=True)
        plot_topo(change, cmin, cmax, drop_ch=bad_ch, cmap='RdBu_r', figname=figname[2], showcolorbar=True)
        plot_topo(rela_change, rmin, rmax, drop_ch=bad_ch, cmap='RdBu_r', figname=figname[3], showcolorbar=True)
        # break

def plot_power_change(acqpre_rp_csv, acq_rp_csv, bad_ch, label):
    onerunrppre1, onerunrp1, change1, rela_change1 = _cal_relapower(acqpre_rp_csv, acq_rp_csv, label='pre1')
    onerunrppre3, onerunrp3, change3, rela_change3 = _cal_relapower(acqpre_rp_csv, acq_rp_csv, label='post3')
    rc = rela_change3 - rela_change1
    vmin, vmax = -0.473, 0.473
    # _print_minmax(rc, bad_ch)
    plot_topo(rc, vmin, vmax, drop_ch=bad_ch, cmap='RdBu_r', figname=label+'_ERDchange', showcolorbar=True)

def plot_power_p(acqpre_rp_csv, acq_rp_csv, bad_ch, label):
    onerunrppre1, onerunrp1, change1, rela_change1 = _cal_relapower(acqpre_rp_csv, acq_rp_csv, label='pre1', ismean=False)
    onerunrppre3, onerunrp3, change3, rela_change3 = _cal_relapower(acqpre_rp_csv, acq_rp_csv, label='post3', ismean=False)
    p = pvalue_2D(rela_change1, rela_change3, ispaired=True)
    plot_topo(p, plot_p=True, drop_ch=bad_ch, figname=label + '_ERD_Pvalue', showcolorbar=True)


def _cal_relapower(pre_csv, csv, label, ismean=True):
    onerunpre = pre_csv[pre_csv['Run'] == label]
    onerun = csv[csv['Run'] == label]
    if ismean:
        onerunrppre = onerunpre.mean().drop(['Subject_id'], axis=0)
        onerunrp = onerun.mean().drop(['Subject_id'], axis=0)
    else:
        onerunrppre = onerunpre.drop(['Subject_name', 'Subject_id', 'Run'], axis=1)
        onerunrp = onerun.drop(['Subject_name', 'Subject_id', 'Run'], axis=1)
    change = onerunrp - onerunrppre
    rela_change = change / onerunrppre  # ERD
    return onerunrppre, onerunrp, change, rela_change


def _print_minmax(data, bad_ch):
    # 计算data的极大 极小值
    data.drop(labels=bad_ch, inplace=True)
    print('min:', round(data.min(), 3), ' max:', round(data.max(), 3))


def _load_minmax(rhythm):
    if rhythm == 'delta':
        vmin, vmax = 0.292, 0.747
        cmin, cmax = -0.075, 0.075
        rmin, rmax = -0.195, 0.195
    elif rhythm == 'theta':
        vmin, vmax = 0.108, 0.191
        cmin, cmax = -0.022, 0.022
        rmin, rmax = -0.123, 0.123
    elif rhythm == 'alpha':
        vmin, vmax = 0.028, 0.267  # rpower
        cmin, cmax = -0.08, 0.08   # change(post-pre)
        rmin, rmax = -0.535, 0.535  # rela change((post-pre)/pre)
    elif rhythm == 'beta11':
        vmin, vmax = 0.023, 0.128
        cmin, cmax = -0.011, 0.011
        rmin, rmax = -0.143, 0.143
    elif rhythm == 'beta12':
        vmin, vmax = 0.023, 0.08
        cmin, cmax = -0.01, 0.01
        rmin, rmax = -0.193, 0.193
    elif rhythm == 'beta13':
        vmin, vmax = 0.015, 0.045
        cmin, cmax = -0.007, 0.007
        rmin, rmax = -0.246, 0.246
    elif rhythm == 'beta21':
        vmin, vmax = 0.036, 0.168
        cmin, cmax = -0.014, 0.014
        rmin, rmax = -0.094, 0.094
    elif rhythm == 'beta22':
        vmin, vmax = 0.027, 0.084
        cmin, cmax = -0.01, 0.01
        rmin, rmax = -0.198, 0.198
    elif rhythm == 'gamma':
        vmin, vmax = 0.033, 0.096
        cmin, cmax = -0.017, 0.017
        rmin, rmax = -0.205, 0.205
    else:
        vmin, vmax = 0, 0
        cmin, cmax = 0, 0
        rmin, rmax = 0, 0
    return vmin, vmax, cmin, cmax, rmin, rmax


def _load_minmax_bylabel(label):
    if label == 'Right':
        vmin, vmax = 0.027, 0.538  # Right/preRight
        cmin, cmax = -0.305, 0.305  # change
        rmin, rmax = -0.58, 0.58  # relative change
    elif label == 'Left':
        vmin, vmax = 0.027, 0.535  # Left/preLeft
        cmin, cmax = -0.288, 0.288  # change
        rmin, rmax = -0.556, 0.556  # relative change
    elif label == 'Rest':
        vmin, vmax = 0.024, 0.539  # Rest/preRest
        cmin, cmax = -0.292, 0.292  # change
        rmin, rmax = -0.581, 0.581  # relative change
    else:
        vmin, vmax = 0, 0
        cmin, cmax = 0, 0
        rmin, rmax = 0, 0
    return vmin, vmax, cmin, cmax, rmin, rmax


if __name__ == '__main__':
    ab_path = r'D:\Myfiles\MIBCI_NF\analysis_df'
    # draw_acc()
    # group_pipeline()
    # baseline_power()
    # neuroFB_power()
    Acq_power()