import numpy as np
from bearing_model import Measurement
from src.utils.reading_and_writing import get_simulation_properties

class PyBearingDatasest():
    def __init__(self, n_severities,failure_modes,quick_iter = False):
        self.simulation_properties = get_simulation_properties(quick_iter=quick_iter)
        # self.n_samples_test = n_samples_test
        # self.n_samples_train = n_samples_train
        self.n_severities = n_severities
        self.failure_modes = failure_modes

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

    def make_measurements_for_different_failure_mode(self, properties_to_modify=None):
        """ For different failure modes, compute many samples for each severity"""

        if properties_to_modify is None:
            properties_to_modify = {}
        failure_mode_dict = {}

        # Instantiate the dictionary of datasets that will follow the structure as in data_set_format_for_project.md

        # Include the healthy data (train and test set)

        # healthy_test_train = self.make_measurements_for_healthy()
        # failure_mode_dict["healthy"] = healthy_test_train
        #  print("Healthy data generation complete in {} min".format((time.time()-t)/60))

        # Include the data for each of the failure modes
        for failure_mode in self.failure_modes:
            properties_to_modify["fault_type"] = failure_mode
            failure_mode_dict[failure_mode] = self.make_measurements_for_different_severity(properties_to_modify)
        return failure_mode_dict
