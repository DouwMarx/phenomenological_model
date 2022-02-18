from pypm.phenomenological_bearing_model.bearing_model import Measurement,BearingData
from pypm.utils.reading_and_writing import get_simulation_properties
import os
import plotly.graph_objs as go
from definitions import plots_dir

spec_dict = get_simulation_properties()


spec_dict.update({
    "measurement_noise_standard_deviation": 0,
    "transient_amplitude_standard_deviation": 0,
})

# m = Measurement(**spec_dict)
m = BearingData(**spec_dict)
measurements = m.get_measurements()
signal = measurements[0]
time = m.time

def main():
    return m