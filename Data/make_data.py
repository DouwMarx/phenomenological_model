import bearing_model
import numpy as np
import yaml
import definitions
from definitions import root_dir
from bearing_model import Measurement
import matplotlib.pyplot as plt
from src.utils.reading_and_writing import get_simulation_properties
from scipy.signal import hilbert
from scipy.signal import detrend


class PyBearingDatasest():
    def __init__(self, n_samples_test, n_samples_train, n_severities,failure_modes):
        self.simulation_properties = get_simulation_properties()
        self.n_samples_test = n_samples_test
        self.n_samples_train = n_samples_train
        self.n_severities = n_severities
        self.failure_modes = failure_modes

        # # Set up the system for the healthy case
        # self.sys_properties_healthy = sys_properties
        # self.sys_properties_healthy["fault_information"]["fault_severity"] = 0
        # self.sys_properties_healthy["fault_information"]["q_amp_mod"] = 0


    # fs = simulation_properties["sampling_frequency"]

    def make_measurements_for_condition(self, properties_to_modify):
        """
        Modify the default values with something

        Parameters
        ----------
        properties_to_modify

        Returns
        -------

        """

        modified_simulation_properties = self.simulation_properties.copy()

        modified_simulation_properties.update(properties_to_modify)

        measurement_obj = Measurement(**modified_simulation_properties)
        meas = measurement_obj.get_measurements()
        return meas

    def make_measurements_for_different_severity(self, properties_to_modify):
        severity_range = np.linspace(0,1,self.n_severities)

        """ For different severities of the same failure mode, compute many samples for each severity"""
        severity_dict = {}
        for degree, severity in enumerate(severity_range):  # Note that the first severity is technically healthy
            properties_to_modify["fault_severity"] = severity
            data = self.make_measurements_for_condition(properties_to_modify)
            severity_dict[str(degree)] = data
        return severity_dict


    # def make_measurements_for_healthy(self):
    #     """ Make training and test set for healthy data """
    #
    #     test_train_dict = {}
    #
    #     for test_train,n_samples in zip(["test", "train"],[self.n_samples_test,self.n_samples_train]):
    #         test_train_dict[test_train] = self.make_many_measurements_for_condition(self.sys_properties_healthy,n_samples)
    #
    #     return test_train_dict

    def make_measurements_for_different_failure_mode(self,properties_to_modify):
        """ For different failure modes, compute many samples for each severity"""

        failure_mode_dict = {}  # Instantiate the dictionary of datasets that will follow the structure as in data_set_format_for_project.md

        # Include the healthy data (train and test set)

        # healthy_test_train = self.make_measurements_for_healthy()
        # failure_mode_dict["healthy"] = healthy_test_train
        #  print("Healthy data generation complete in {} min".format((time.time()-t)/60))

        # Include the data for each of the failure modes
        for failure_mode in self.failure_modes:
            properties_to_modify["fault_type"] = failure_mode
            failure_mode_dict[failure_mode] = self.make_measurements_for_different_severity(properties_to_modify)
            # print(failure_mode + " data generation complete in {} min".format((time.time()- t) / 60))
        return failure_mode_dict


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

def sq_env_spec(signal,fs):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = detrend(amplitude_envelope)

    freq, mag, phase = fft(amplitude_envelope, fs)

    return freq,mag,phase


o = PyBearingDatasest(n_samples_test=4,n_samples_train=4,n_severities=3,failure_modes=["ball","inner"])
properties_to_modify = {"fault_type":"outer"}

# r = o.make_measurements_for_condition(properties_to_modify)

# d = o.make_measurements_for_different_severity(properties_to_modify)
# print(d.keys())

da = o.make_measurements_for_different_failure_mode(properties_to_modify)


# plt.figure()
# for key,val in d.items():
#     plt.plot(val[0],label = key)
# plt.legend()