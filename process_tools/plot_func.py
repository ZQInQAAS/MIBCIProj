import mne
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from scipy import stats
from BCIConfig import ch_names60, ch_types60, fs


def wilcoxon_p(x, y):
    # wilcoxon rank sum 秩和检验
    res = stats.mannwhitneyu(x, y)
    return res.pvalue


def pvalue_1D(groupA, groupB, ispaired=False):
    # group_A/group_B (subject_num,) p: float
    p = pg.ttest(groupA, groupB, paired=ispaired)['p-val'].values[0]
    # p = pg.wilcoxon(group_A, group_B)
    return p


def pvalue_2D(group_A, group_B, ispaired=False):
    # 计算每个ch上 前后的组间差异
    # group_A/group_B df(subject_num, ch) p_list:(ch,)
    p_list = pd.Series(index=group_A.columns)
    for i in range(group_A.shape[1]):
        # p = wilcoxon_rank_sum_test(group_A[:, i], group_B[:, i])
        p = pvalue_1D(group_A.iloc[:, i], group_B.iloc[:, i], ispaired=ispaired)
        p_list.iloc[i] = p
    reject, p_cor = pg.multicomp(np.array(p_list), method='fdr_bh')  # 多重比较验证
    return p_list


def pvalue_3D(group_A, group_B, ispaired=False):
    # 计算conv矩阵每个ch对 前后的组间差异
    # group_A/group_B array(subject_num, ch, ch) p_list:(ch, ch)
    n_ch = group_A.shape[1]
    p_conv = np.zeros([n_ch, n_ch])
    for i in range(n_ch):
        for j in range(i):
            # p = wilcoxon_rank_sum_test(group_A[:, i, j], group_B[:, i, j])
            p = pvalue_1D(group_A[:, i, j], group_B[:, i, j], ispaired=ispaired)
            p_conv[i, j] = p
    reject, p_cor = pg.multicomp(p_conv.flatten(), method='fdr_bh')  # 多重比较验证
    return p_conv


def plot_group_boxplot(group_df):
    # 组间差异及显著性 箱线图 group_df(每列为一组)
    plt.figure(figsize=(4, 5))
    plt.rcParams["axes.labelsize"] = 15
    a = sns.boxplot(data=group_df, palette="Set2", width=0.5)
    randomLevel = 0.3333  # 0.3333
    a.axhline(randomLevel, ls='--', c='gray', lw=1.3)  # 随机水平
    # lim = _get_lim(group_df)
    # plt.ylim(lim)
    a.set_ylim([0.2, 1.05])
    a.tick_params(labelsize=15)
    # 绘制显著性注释
    x1, x2 = 0, 1  # 第一第二列
    y, h, color = group_df.stack().max() + 0.03, 0.015, '#6A6A6A'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
    lab = '**'  # 'n.s.'
    plt.text((x1 + x2) * .5, y + h, lab, ha='center', va='bottom', color=color)
    plt.ylabel('Accuracy')
    plt.xlabel('Stage')
    plt.tight_layout()
    plt.show()


def plot_subject_barplot(data, x, y, hue):
    # 并列柱状图 前后变化
    plt.rcParams["axes.labelsize"] = 15
    a = sns.barplot(x, y, hue, data, palette="Set2", errwidth=1.3, capsize=.3)
    randomLevel = 0.3333  # 0.3333
    a.axhline(randomLevel, ls='--', c='gray', lw=1.3)  # 随机水平
    a.set_ylim([0.2, 1.05])
    a.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()


def _get_lim(x, range=0.05):
    try:
        mi, ma = x.stack().min(), x.stack().max()  # dataframe
    except AttributeError:
        mi, ma = x.min(), x.max()  # series
    d = (ma - mi) * range
    return mi - d, ma + d


def plot_regplot(x, y, xlabel='', ylabel=''):
    # 绘制线性相关散点图(拟合)
    sns.set_palette(sns.color_palette("Set2"))
    plt.rcParams["axes.labelsize"] = 15
    c = sns.regplot(x, y)
    c.tick_params(labelsize=15)
    xlim, ylim = _get_lim(x), _get_lim(y)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # c.set_ylim([0.539, 0.9268])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_topo(data, vmin=None, vmax=None, cmap='Reds', drop_ch=None, plot_p=False, showcolorbar=False, figname=''):
    # 2D脑拓扑图  data 1D Series (ch,)
    if drop_ch:
        data.drop(labels=drop_ch, inplace=True)
        if len(ch_names60) != len(data):
            for i in drop_ch:
                ch_names60.remove(i)
                ch_types60.pop(1)
    info = mne.create_info(ch_names60, fs, ch_types60)
    info.set_montage('standard_1005')
    if vmin is None and vmax is None:
        if plot_p:
            vmin, vmax = 0, 0.1
        else:
            vmin, vmax = data.min(), data.max()
    # p值: 白红黑'hot'  原值: 红白'Reds' 差值: 红蓝色'RdBu_r'
    cmap = 'hot' if plot_p else cmap
    # plt.tight_layout()
    fig = plt.figure()
    name_list = _topo_labellist(len(data), False)
    # data, cmap = plot_legend(len(data))
    im, _ = mne.viz.plot_topomap(data, info, vmin=vmin, vmax=vmax, cmap=cmap,
                                 names=name_list, show_names=False, contours=0, show=False)
    fig.savefig(r'figure/' + figname + '.png')
    if showcolorbar:
        position = fig.add_axes([0.84, 0.11, 0.03, 0.8])  # 位置[左,下,右,上] 一位小数0.84 两位小数0.82
        c = plt.colorbar(im, fraction=0.04, cax=position)
        c.ax.tick_params(labelsize=20)
        plt.savefig(r'figure/' + figname + '_bar.png')
        # plt.show()


def plot_legend(ch=56):
    data = np.array([0, ] * ch)
    cmap = 'RdBu_r'
    return data, cmap


def _topo_labellist(ch, islegend=False):
    name_list = ['', ] * ch
    if islegend:
        name_list[1] = 'Fpz'
        name_list[9] = 'Fz'
        name_list[24] = 'C3'
        name_list[26] = 'Cz'
        name_list[28] = 'C4'
        name_list[41] = 'Pz'
        name_list[54] = 'Oz'
    return name_list


if __name__ == '__main__':
    pass
