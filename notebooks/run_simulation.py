import numpy as np
import yaml

import definitions
from definitions import root_dir
from bearing_model import Measurement
import matplotlib.pyplot as plt

with open(root_dir.joinpath("simulation_properties.yml"), "r") as file:
    yaml_properties = yaml.safe_load(file)


def flatten_dict(dict):
    """
    Extracts the leafs of a dictionary with 2 levels
    Parameters
    ----------
    dict

    Returns
    -------

    """
    return {key: val for sub_dict in dict.values() for key, val in sub_dict.items()}


# simulation_properties = {"speed_profile_type": "constant"}
simulation_properties = flatten_dict(yaml_properties)
measurement_obj = Measurement(**simulation_properties)

meas = measurement_obj.get_measurements()

np.save(definitions.data_write_dir,meas)

plt.figure()
# plt.plot(indexes_at_which_impulses_occur[0]*np.max(convolved[0]))
plt.plot(meas[0])

