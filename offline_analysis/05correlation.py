import os
import mne
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from process_tools import plot_regplot
from BCIConfig import ch_names60, pick_rest_ch
from pingouin import corr, rm_corr, plot_rm_corr, pairwise_corr


def pgcorr(x, y, method='spearman'):
    pea = corr(x, y, method=method)
    r, p = pea['r'].values[0], pea['p-val'].values[0]
    return r, p


def person_cor():
    rpower = pd.read_csv(r'analysis_df\rp\df_power_Bas_alpha_60s_1.csv')
    acc0 = pd.read_csv(r'analysis_df\Acc_raw\df_class123_tenfold_Acc_60ch_raw.csv')
    acc = acc0.loc[:, ['Subject_id', 'Accuracy12']]
    # df = pd.merge(acc, rpower)
    df = pd.concat([acc, rpower], axis=1)
    g = df.groupby(df['Subject_name']).mean()
    g = g.drop(['Subject_id', 'Session', ], axis=1)
    # g1 = g.drop([4,8,7,9,17 ], axis=0)
    stats = pairwise_corr(g, columns=[["Accuracy12", ], None])  # (ch_num, n)
    stat = stats[stats['X'] == 'Accuracy12']
    r = stat.loc[:, ['r']].values.squeeze()
    p = stat.loc[:, ['p-unc']].values.squeeze()
    # p_c = stat.loc[:, ['p-corr']].values.squeeze()  # Corrected p-values.
    print('Correlation coefficients:', r, 'Uncorrected p-values:', p)
    p[np.where(p > 0.1)] = 0.1
    # stats.to_csv('pairwise_corr_alpha.csv', index=False)
    return r, p


def drop_sub(data_df, sub_list, col_name='Subject_name'):
    # 移除部分被试
    data_df_sub = data_df.drop(data_df[data_df[col_name].isin(sub_list)].index)
    return data_df_sub


def cal_corr_pip():
    # 表现和RP在NFT前后的变化是否有相关性及显著性
    ap = r'D:\Myfiles\MIBCI_NF\analysis_df'
    bad_sub = ['LYR', 'CYH', 'LXT', 'FX']

    acc_csv = pd.read_csv(ap + r'\Acc_raw\df_classLR_tenfold_Acc_60ch_raw.csv')
    acc_csv_sub = drop_sub(acc_csv, bad_sub, 'Subject_name')
    acc_m = acc_csv_sub.groupby(acc_csv_sub['Subject_id']).mean()
    # acc_change = change(acc_m)
    acc_change = acc_m['Post'] - acc_m['Pre']
    # print(np.mean(acc_m['Pre']), np.mean(acc_m['Post']))  # raw 0.593 0.648  去4ch 0.591 0.652
    acc123_csv = pd.read_csv(ap + r'\Acc_raw\df_class123_tenfold_Acc_60ch_raw.csv')
    acc123_csv_sub = drop_sub(acc123_csv, bad_sub, 'Subject_name')
    acc123_m = acc123_csv_sub.groupby(acc123_csv_sub['Subject_id']).mean()
    acc123_change = acc123_m['Post'] - acc123_m['Pre']
    # acc123_change = change(acc123_m)

    # mrt_score_csv = pd.read_csv(ap + r'\mrt_score.csv')
    # mrt_score_csv_sub = drop_sub(mrt_score_csv, bad_sub, 'Subject_name')
    # mrt_m = mrt_score_csv_sub.groupby(mrt_score_csv_sub['Subject_id']).mean()

    rpower_csv = pd.read_csv(ap + r'\Baseline_power\df_rela_power_Bas_alpha_60s_1.csv')
    rpower_csv_sub = drop_sub(rpower_csv, bad_sub, 'Subject_name')
    rpower_pre_eo = rpower_csv_sub[rpower_csv_sub['Run'] == 'pre_eo']
    pre_eo = rpower_pre_eo.groupby(rpower_pre_eo['Subject_id']).mean()
    rpower_post_eo = rpower_csv_sub[rpower_csv_sub['Run'] == 'post_eo']
    post_eo = rpower_post_eo.groupby(rpower_post_eo['Subject_id']).mean()
    pre_eo_mean = np.mean(pre_eo[pick_rest_ch], axis=1)
    post_eo_mean = np.mean(post_eo[pick_rest_ch], axis=1)
    rp = (post_eo_mean - pre_eo_mean)/pre_eo_mean
    # cal r, p
    r, p = pgcorr(acc_change, acc123_change, method='pearson')
    print('r=', round(r, 3), 'p=', round(p, 3))
    # r, p = pgcorr(acc_change, mrt_m['Change'], method='pearson')
    # print('r=', round(r, 3), 'p=', round(p, 3))
    # r, p = pgcorr(mrt_m['Change'], rp, method='pearson')
    # print('r=', round(r, 3), 'p=', round(p, 3))
    # plot
    plot_regplot(acc_change, acc123_change, xlabel='accuracy', ylabel='relative power')
    # plot_regplot(acc_change, mrt_m['Change'], xlabel='accuracy', ylabel='MRT score')
    # plot_regplot(rp, mrt_m['Change'], xlabel='relative power', ylabel='MRT score')


def change(df):
    pre = df.loc[:, ['Pre1', 'Pre2', 'Pre3']].mean(axis=1)
    post = df.loc[:, ['Post1', 'Post2', 'Post3']].mean(axis=1)
    change = post - pre
    return change


if __name__ == '__main__':
     cal_corr_pip()  # raw 60ch r=0.65 p=0.004 去4ch r=0.601 p=0.008