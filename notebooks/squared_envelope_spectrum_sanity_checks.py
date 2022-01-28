import numpy as np
from src.data.phenomenological_bearing_model.bearing_model import Measurement
import matplotlib.pyplot as plt
from src.utils.reading_and_writing import get_simulation_properties
from src.utils import sigproc

simulation_properties = get_simulation_properties(quick_iter=True)  # Load a dictionary of simulation properties

# Define a bearing measurement object and compute the measurements
measurement_obj = Measurement(**simulation_properties)
measurements = measurement_obj.get_measurements()

# Compute and plot the signal envelope
amplitude_envelope = sigproc.envelope(measurements)

plt.figure()
plt.plot(measurement_obj.measured_time, measurements[0])
plt.plot(measurement_obj.measured_time, amplitude_envelope[0])


# Compute and plot the envelope spectrum
fs = simulation_properties["sampling_frequency"]
freq,mag,phase = sigproc.env_spec(measurements, fs)

plt.figure()
plt.plot(freq, mag[0])

impulse_per_rev = measurement_obj.meta_data["derived"]["geometry_factor"]

fault_freq = measurement_obj.meta_data["derived"]["average_fault_frequency"]
plt.vlines(np.arange(1, 6)*fault_freq, 0, np.max(mag),"k")


