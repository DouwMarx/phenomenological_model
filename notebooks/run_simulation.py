import numpy as np
import yaml

import definitions
from definitions import root_dir
from bearing_model import Measurement
import matplotlib.pyplot as plt
from src.utils.reading_and_writing import get_simulation_properties
from scipy.signal import hilbert
from scipy.signal import detrend


def fft(data, fs):
    """
    Parameters
    ----------
    data: String
        The heading name for the dataframe
    Returns
    -------
    freq: Frequency range
    magnitude:
    phase:
    """

    d = data
    length = len(d)
    Y = np.fft.fft(d) / length
    magnitude = np.abs(Y)[0:int(length / 2)]
    phase = np.angle(Y)[0:int(length / 2)]
    freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]
    return freq, magnitude, phase


simulation_properties = get_simulation_properties()
measurement_obj = Measurement(**simulation_properties)

meas = measurement_obj.get_measurements()

# np.save(definitions.data_write_dir,{"data":meas,"meta_data":simulation_properties})

example = meas[0]

analytic_signal = hilbert(example)
amplitude_envelope = np.abs(analytic_signal)
amplitude_envelope = detrend(amplitude_envelope)

plt.figure()
# plt.plot(indexes_at_which_impulses_occur[0]*np.max(convolved[0]))
plt.plot(measurement_obj.time[::2],meas[0])
plt.plot(measurement_obj.time[::2],amplitude_envelope)


fs = simulation_properties["sampling_frequency"]
freq,mag,phase = fft(amplitude_envelope,fs)
plt.figure()
plt.plot(freq,mag)

impulse_per_rev = measurement_obj.meta_data["derived"]["geometry_factor"]
fr = np.mean(measurement_obj.get_rotation_frequency_as_function_of_time())/(2*np.pi) # revs/s

fault_freq =impulse_per_rev*fr

plt.vlines(fault_freq,0,np.max(mag),"k")




