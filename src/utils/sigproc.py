import numpy as np
from scipy.signal import hilbert
from scipy.signal import detrend


def fft_mag(data, fs):
    """
    Computes the magnitude and phase for positive frequencies. Fourier transform .
    """

    length = data.shape[1]
    Y = np.fft.fft(data, axis=1) / length
    magnitude = np.abs(Y)[:, 0:int(length / 2)]
    phase = np.angle(Y)[:, 0:int(length / 2)]
    freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]
    return freq, magnitude, phase


def env_spec(signals, fs):
    """
    Compute the envelope spectrum
    Parameters
    ----------
    signals
    fs

    Returns
    -------

    """
    amplitude_envelope = envelope(signals)
    freq, mag, phase = fft_mag(amplitude_envelope, fs)
    return freq, mag, phase


def envelope(array):
    """
    Compute the time domain envelope of the signal
    Parameters
    ----------
    array

    Returns
    -------

    """
    ana = hilbert(array, axis=1)
    amplitude_envelope = np.abs(ana)
    # print(np.isnan(ana).sum())
    return amplitude_envelope
