from scipy.signal import cheby2, filtfilt


def bandpass_filter(data, fs, filter_low, filter_high):
    wn1 = filter_low / (fs / 2)
    wn2 = filter_high / (fs / 2)
    [b, a] = cheby2(N=6, rs=40, Wn=[wn1, wn2], btype='bandpass')
    after_filter_x = filtfilt(b, a, data, axis=0)
    return after_filter_x
