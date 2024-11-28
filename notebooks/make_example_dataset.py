import numpy as np
from pypm.phenomenological_bearing_model.make_data import  PyBearingDataset
from definitions import data_dir

o = PyBearingDataset(n_severities=5, failure_modes=["ball", "inner", "outer"],quick_iter=False,parallel_evaluate=True)
props_to_modify = {"t_duration":1,
                   "n_measurements":20}
m = o.make_measurements_for_different_failure_mode(properties_to_modify=props_to_modify)

np.save(data_dir.joinpath("example_dataset"), m, allow_pickle=True)


def main():
    return m