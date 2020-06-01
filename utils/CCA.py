import numpy as np
from scipy.io import loadmat, savemat


def bsscca_ifc(data_x, remove_idx=None):
    """
    data_x/data_x_cca: eeg signals (n_ch, n_samples)
    W: 分离矩阵 (f, n_ch)  默认 f==n_ch
    A: 混合矩阵 (n_ch, f) --W的伪逆矩阵
    s: source(component) signals (f, n_samples)
    remove_idx: 要移除(置0)的伪迹成分idx  0低频(眼电/慢漂移) -1高频(肌电/白噪)
    """
    if remove_idx is None:
        remove_idx = [0, -1]
    Wpca, Xpca = pca(data_x)
    W, r = bsscca(Xpca)
    W = np.dot(W, Wpca)
    A = np.linalg.pinv(W)
    s = np.dot(W, data_x)
    s[remove_idx, :] = 0
    data_x_cca = np.dot(A, s)
    return data_x_cca


def pca(data_x, num_sources=None, eigratio=1e6):
    """
    num_sources: 源的数量(默认与n_channel相同)
    eigratio: 数据协方差特征值的最大扩展，以lambda_max / lambda_min来度量
    """
    num_sources = num_sources if num_sources else data_x.shape[0]
    C = np.cov(data_x)
    D, V = np.linalg.eig(C)
    I = np.argsort(-abs(D))
    val = D[I]
    while val[0] / val[num_sources-1] > eigratio:
        num_sources = num_sources - 1
    V = V[:, I[:num_sources]]
    t = D[I[:num_sources]]
    D = np.diag(t**(-0.5))
    a = V.T
    W = np.dot(D, a)
    Y = np.dot(W, data_x)
    return W, Y


def bsscca(data_x, delay=1):
    """
    Blind Source Separation through Canonical Correlation Analysis 典型相关分析实现盲源分离
    data_x - data matrix 2D(channal, sample)
    delay  - delay at which the autocorrelation of the sources will be maximized (def: 1) CCA估计的延迟
    W     - separation matrix 分离矩阵
    r     - autocorrelation of the estimated sources at the given delay 估计源在给定延迟的自相关
    """
    T = len(data_x[0])
    Y = data_x[:, delay:]
    data_x = data_x[:, 0:-delay]
    Cyy = np.dot(np.dot((1 / T), Y), Y.T)  # 自协方差矩阵
    Cxx = np.dot(np.dot((1 / T), data_x), data_x.T)
    Cxy = np.dot(np.dot((1 / T), data_x), Y.T)  # 互协方差矩阵
    Cyx = Cxy.T
    invCxx, invCyy = np.linalg.pinv(Cxx), np.linalg.pinv(Cyy)
    r, W = np.linalg.eig(np.dot(np.dot(np.dot(invCxx, Cxy), invCyy), Cyx))
    r = np.sqrt(abs(np.real(r)))
    I = np.argsort(-r)
    W = W[:, I].T
    return W, r


if __name__ == '__main__':
    m = loadmat(r'D:\Myfiles\EEGProject\data_set\data_set_public\BCICompetitionIV\2amat\A01T.mat')
    x = m['data_x'][:, :, 0].T
    x = np.array(x, dtype=np.float)
    data_x_cca = bsscca_ifc(x)
    print(data_x_cca)
