import numpy as np
import yaml

import definitions
from definitions import root_dir
from bearing_model import Measurement
import matplotlib.pyplot as plt
from src.utils.reading_and_writing import get_simulation_properties




simulation_properties = get_simulation_properties()
measurement_obj = Measurement(**simulation_properties)

meas = measurement_obj.get_measurements()

np.save(definitions.data_write_dir,{"data":meas,"meta_data":simulation_properties})

# plt.figure()
# # plt.plot(indexes_at_which_impulses_occur[0]*np.max(convolved[0]))
# plt.plot(meas[0])

