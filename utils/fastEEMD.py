import os
import numpy as np
import matlab.engine  # 仅支持python36以下


def eemd(signal=None, NoiseLevel=0.4, NE=5):
    """
    fast ensemble empirical mode decomposition 快速集成经验模态分解
    input signal: 1D
    NE: ensemble number
    numImf: number of prescribed imf(intrinsic mode functions)
    """
    eng = matlab.engine.start_matlab()
    path = os.path.dirname(os.getcwd()) + r'\utils'
    eng.cd(path, nargout=0)
    numImf = 8  # 同步修改emdAPI.m参数
    xsize = len(signal)
    Ystd= np.std(signal)
    allmode = np.zeros([xsize, numImf])
    for i in range(NE):
        np.random.seed(i)  # random seed number in generating white noise 白噪声
        temp = np.multiply(np.dot(np.random.randn(1, xsize), NoiseLevel), Ystd)
        xend = matlab.double((signal + temp)[0].tolist())
        imf = eng.emdAPI(xend)
        allmode = allmode + np.array(imf)
    allmode = allmode / NE
    allmode = allmode.T
    return allmode


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    m = loadmat(r'D:\Myfiles\EEGProject\data_set\data_set_public\BCICompetitionIV\2amat\A01T.mat')
    x = m['data_x'][:, 0, 0].T
    allmode = eemd(x)
    # savemat('a1.mat', {'a': allmode})
