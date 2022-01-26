import numpy as np
from bearing_model import Measurement
import matplotlib.pyplot as plt
from src.utils.reading_and_writing import get_simulation_properties
from scipy.signal import hilbert
# from scipy.fftpack import fft,ifft
# def custom_hilbert(x, N=None, axis=-1):
#     Xf = fft(x, N, axis=axis)
#     h = np.zeros(N)
#     if N % 2 == 0:
#         h[0] = h[N // 2] = 1
#         h[1:N // 2] = 2
#     else:
#         h[0] = 1
#         h[1:(N + 1) // 2] = 2
#
#     if x.ndim > 1:
#         ind = [np.newaxis] * x.ndim
#         ind[axis] = slice(None)
#         h = h[tuple(ind)]
#     x = ifft(Xf * h, axis=axis)
#     return x


def fft_mag(data, fs):
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

    length = data.shape[1]
    Y = np.fft.fft(data,axis=1) / length
    magnitude = np.abs(Y)[:,0:int(length / 2)]
    phase = np.angle(Y)[:,0:int(length / 2)]
    freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]
    return freq, magnitude, phase

def sq_env_spec(signals,fs):

    analytic_signal = hilbert(signals,axis=1)
    amplitude_envelope = np.abs(analytic_signal)
    # amplitude_envelope = detrend(amplitude_envelope,type="constant",axis=1)

    freq, mag, phase = fft_mag(amplitude_envelope, fs)
    return freq,mag,phase

def envelope(array):
    # print(np.isnan(array).sum())
    ana = hilbert(array,axis=1)
    # print(np.isnan(ana).sum())
    amplitude_envelope =np.abs(ana)
    # print(np.isnan(amplitude_envelope).sum())
    return amplitude_envelope


class PyBearingDatasest():
    def __init__(self, n_severities,failure_modes):
        self.simulation_properties = get_simulation_properties()
        # self.n_samples_test = n_samples_test
        # self.n_samples_train = n_samples_train
        self.n_severities = n_severities
        self.failure_modes = failure_modes


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

        # TODO: Need to add functionality that checks that the properties that is being modified is actually in the dictionary. Otherwise you might think you changed something but did not

        # Make a copy of the default simulation properties
        modified_simulation_properties = self.simulation_properties.copy()

        # Update the simulation properties with the properties that are of interest for the current simulation
        modified_simulation_properties.update(properties_to_modify)


        # Define the measurement

        mobj = Measurement(**modified_simulation_properties)
        meas = mobj.get_measurements() # Get the time_domain measurements

        np.linalg.norm(meas)
        # np.mean(meas)# = meas
        # print(np.linalg.norm(meas))
        # np.mean(meas)# = meas

        # Compute the squared envelope spectrum
        # freq,mag,phase = sq_env_spec(meas,fs=self.simulation_properties["sampling_frequency"])
        # env = envelope(meas)

        # Create a dictionary with different flavours of the same data as well as meta data

        modified_simulation_properties.update(mobj.meta_data) # Add derived meta-data to the meta data for later use # TODO: build this functionality into the bearing class

        meas_dict = {"time_domain": meas,
                     # "squared_envelope":{"freq":freq,
                     #                     "mag":mag,
                     #                     "phase":phase},
                     # "envelope":env,
                     "meta_data":modified_simulation_properties}

        return meas_dict
        # return meas

    def make_measurements_for_different_severity(self, properties_to_modify):
        severity_range = np.linspace(0, 1, self.n_severities)

        """ For different severities of the same failure mode, compute many samples for each severity"""
        severity_dict = {}
        for degree, severity in enumerate(severity_range):  # Note that the first severity is technically healthy
            modified_properties = properties_to_modify.copy()
            modified_properties["fault_severity"] = severity
            data = self.make_measurements_for_condition(modified_properties)
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
        return failure_mode_dict



# TODO: The generation of the envelope spectrum using the hilbert transform delivers differernt results for consequent runs and I cannot understand it
    # o = PyBearingDatasest(n_samples_test=4, n_samples_train=4, n_severities=3, failure_modes=["ball","inner"])
    # properties_to_modify = {"fault_type":"inner","fault_severity":1}
    # r = o.make_measurements_for_condition(properties_to_modify)
    # print(np.isnan(r["envelope"]).sum())
    #
    # ob = PyBearingDatasest(n_samples_test=4, n_samples_train=4,n_severities=3,failure_modes=["ball","inner"])
    # properties_to_modify = {"fault_type":"inner","fault_severity":1}
    # rb = ob.make_measurements_for_condition(properties_to_modify)
    # print(np.isnan(rb["envelope"]).sum())

o = PyBearingDatasest(n_severities=2, failure_modes=["ball","inner"])
properties_to_modify = {"fault_type":"inner","fault_severity":1}
results_dictionary = o.make_measurements_for_different_failure_mode(properties_to_modify)

plt.figure()

for mode,dictionary in results_dictionary.items():
    print(mode)
    for severity, measurements in dictionary.items():
        t_sig = measurements["time_domain"]
        env = envelope(t_sig)
        print(np.isnan(env).sum())
        freq,mag,phase= sq_env_spec(t_sig,fs=o.simulation_properties["sampling_frequency"])
        print(np.isnan(mag).sum())
        results_dictionary[mode][severity]["envelope"] = env

        plt.plot(freq,mag[0], label = mode + " " + severity)

for mode,dictionary in results_dictionary.items():
    for severity, measurements in dictionary.items():
        meta_data = measurements["meta_data"]
        fault_freq = meta_data["derived"]["average_fault_frequency"]

        plt.vlines(fault_freq,0,0.1, label=mode + " ff")
plt.legend()


