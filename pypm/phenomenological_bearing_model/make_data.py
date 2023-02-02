import numpy as np
from pypm.phenomenological_bearing_model.bearing_model import Measurement, BearingData
from pypm.utils.reading_and_writing import get_simulation_properties, flatten_dict
import pathlib


class PyBearingDataset(object):
    def __init__(self, n_severities=2, failure_modes=["inner","outer","ball"], quick_iter=False, parallel_evaluate=False,simulation_properties=None):
        if simulation_properties is None:
            self.simulation_properties = get_simulation_properties(quick_iter=quick_iter)
            print("Using default simulation properties")
        # If the simulation properties are specified directly, do not use the default
        elif isinstance(simulation_properties, dict):
            self.simulation_properties = flatten_dict(simulation_properties)
        else:
            raise ValueError("simulation_properties should be a dictionary")

        self.n_severities = n_severities
        self.failure_modes = failure_modes
        self.parallel_evaluate = parallel_evaluate

    def make_measurements_for_condition(self, properties_to_modify):
        """
        Modify the default parameters of the simulation as defined in simulation_properties.yml and create a
        measurement from a bearing measurement object

        Parameters
        ----------
        properties_to_modify: A dictionary of parameters that should be different from the default definition.

        Returns
        -------

        """

        # TODO: Need to add functionality that checks that the properties that is being modified is actually in the
        #  dictionary. Otherwise you might think you changed something but did not

        # Make a copy of the default simulation properties
        modified_simulation_properties = self.simulation_properties.copy()

        # Update the simulation properties with the properties that are of interest for the current simulation
        modified_simulation_properties.update(properties_to_modify)

        # Define the measurement
        measurement_obj = BearingData(**modified_simulation_properties)
        meas = measurement_obj.get_measurements()  # Get the time_domain measurements

        # Create a dictionary with different flavours of the same data as well as meta data

        sampling_frequency = modified_simulation_properties["sampling_frequency"]

        meta_data = {"simulation_governing_parameters":modified_simulation_properties,
                     "sampling_frequency":sampling_frequency}

        meta_data.update({key:val for key,val in measurement_obj.derived_meta_data.items()}) # Add for example the mean rotation frequency

        meas_dict = {"time_domain": meas,
                     "meta_data": meta_data}

        return meas_dict

    def make_measurements_for_different_severity(self, properties_to_modify):
        """
        Loop through severities and create measurements for each of them

        Parameters
        ----------
        properties_to_modify

        Returns
        -------

        """

        severity_range = np.linspace(0, 1, self.n_severities)  # Assumption that severities are scaled between 0 and 1

        severity_dict = {}
        for degree, severity in enumerate(severity_range):  # Note that the first severity (0) is actually healthy
            modified_properties = properties_to_modify.copy()
            modified_properties["fault_severity"] = severity  # Update the severity of the default parameters

            data = self.make_measurements_for_condition(modified_properties)
            severity_dict[str(degree)] = data # Add the generated pypm to a dictionary of the form {"0":pypm,"1":pypm... etc)

        return severity_dict

    # TODO: Consider adding a dedicated function for generating pypm under healthy conditions (ie not severity = 0)

    def make_measurements_for_different_failure_mode(self, properties_to_modify=None):
        """ For different failure modes, compute many samples for each severity"""

        if properties_to_modify is None:
            properties_to_modify = {}

        # Instantiate the dictionary of datasets that will follow the structure as in data_set_format_for_project.md
        # (see anomaly detection project)

        if self.parallel_evaluate:
            from joblib import Parallel, delayed
            def process(failure_mode):
                properties_to_modify["fault_type"] = failure_mode
                return {failure_mode: self.make_measurements_for_different_severity(properties_to_modify)}
            results = Parallel(n_jobs=3)(delayed(process)(failure_mode) for failure_mode in self.failure_modes) # TODO: njobs independent specify

            failure_mode_dict = {key:val for x in results for key,val in x.items()}

        else:
            def process(failure_mode):
                properties_to_modify["fault_type"] = failure_mode
                return {failure_mode: self.make_measurements_for_different_severity(properties_to_modify)}
            results = [process(failure_mode) for failure_mode in self.failure_modes]
            failure_mode_dict = {key:val for x in results for key,val in x.items()}

        return failure_mode_dict

class LinearSeverityIncreaseDataset(object):
    def __init__(self, n_test_samples, n_healthy_samples, failure_modes, quick_iter=False, parallel_evaluate=False):
        self.n_test_samples = n_test_samples
        self.n_healty_samples = n_healthy_samples
        self.simulation_properties = get_simulation_properties(quick_iter=quick_iter)
        self.simulation_properties.update({"n_measurements":1}) # Only take one measurement per operating condition / severity
        self.failure_modes = failure_modes
        self.parallel_evaluate = parallel_evaluate

    def make_measurements_for_condition(self, properties_to_modify):
        """
        Modify the default parameters of the simulation as defined in simulation_properties.yml and create a
        measurement from a bearing measurement object

        Parameters
        ----------
        properties_to_modify: A dictionary of parameters that should be different from the default definition.

        Returns
        -------

        """

        # TODO: Need to add functionality that checks that the properties that is being modified is actually in the
        #  dictionary. Otherwise you might think you changed something but did not

        # Make a copy of the default simulation properties
        modified_simulation_properties = self.simulation_properties.copy()

        # Update the simulation properties with the properties that are of interest for the current simulation
        modified_simulation_properties.update(properties_to_modify)

        # Define the measurement
        measurement_obj = BearingData(**modified_simulation_properties)
        meas = measurement_obj.get_measurements()  # Get the time_domain measurements

        # Create a dictionary with different flavours of the same pypm as well as meta pypm

        sampling_frequency = modified_simulation_properties["sampling_frequency"]

        meta_data = {"simulation_governing_parameters":modified_simulation_properties,
                     "sampling_frequency":sampling_frequency}


        meta_data.update({key:val for key,val in measurement_obj.derived_meta_data.items()}) # Add for example the mean rotation frequency

        meas_dict = {"time_series": list(meas.flatten()),
                     "meta_data": meta_data}

        return meas_dict

    def make_measurements_for_different_severity(self, properties_to_modify):
        """
        Loop through severities for mode and create measurements for each of them
        -------
        """

        severity_range = np.linspace(0, 1, self.n_test_samples+1)[1:]  # Assumption that severities are scaled between 0 and 1
                                                                       # The +1 and [1:] ensures that the first sample here will actually be faulty

        measurements_for_mode_at_severities = []
        for record_number, severity in enumerate(severity_range):  # Note that the first severity (0) is actually healthy
            modified_properties = properties_to_modify.copy()
            modified_properties["fault_severity"] = severity  # Update the severity of the default parameters

            meas_dict = self.make_measurements_for_condition(modified_properties)
            meas_dict.update({"record_number":self.n_healty_samples + record_number+1, # healthy samples are the first records, faulty follow thereafter, 1,based
                              "mode":modified_properties["fault_type"],
                              "severity":int(np.ceil(10*severity/severity_range[-1]))}  # Get severity groups 1-10
                             ) # Add a record number to the measurement use 1-based indexing

            measurements_for_mode_at_severities.append(meas_dict)

        return measurements_for_mode_at_severities

    def make_measurements_for_healthy(self, properties_to_modify):
        """
        Loop through severities for mode and create measurements for each of them
        -------
        """

        properties_to_modify.update({"fault_severity":0}) # All examples from this function should be in healthy condition

        measurements_for_healthy = []
        for record_number in range(1,self.n_healty_samples+1):
            meas_dict = self.make_measurements_for_condition(properties_to_modify)
            meas_dict.update({"record_number":record_number, # healthy samples are the first records, faulty follow thereafter, 1,based
                              "mode":None, # Data is technically not associated with a failure mode 
                              "severity":0}  # Get severity groups 1-10
                             ) # Add a record number to the measurement use 1-based indexing

            measurements_for_healthy.append(meas_dict)

        return measurements_for_healthy

    # TODO: Consider adding a dedicated function for generating pypm under healthy conditions (ie not severity = 0)

    def make_measurements_for_different_failure_mode(self, properties_to_modify=None):
        """ For different failure modes, compute many samples for each severity"""

        if properties_to_modify is None:
            properties_to_modify = {}

        # Instantiate the dictionary of datasets that will follow the structure as in data_set_format_for_project.md
        # (see anomaly detection project)

        if self.parallel_evaluate:
            from joblib import Parallel, delayed
            def process(failure_mode):
                properties_to_modify["fault_type"] = failure_mode
                meas_at_mode_for_all_sev = self.make_measurements_for_different_severity(properties_to_modify)
                return meas_at_mode_for_all_sev # TODO: Need to update the mode here

            results = Parallel(n_jobs=3)(delayed(process)(failure_mode) for failure_mode in self.failure_modes) # TODO: njobs independent specify


        else:
            def process(failure_mode):
                properties_to_modify["fault_type"] = failure_mode
                meas_at_mode_for_all_sev = self.make_measurements_for_different_severity(properties_to_modify)
                return meas_at_mode_for_all_sev # TODO: Need to update the mode here

            results = [process(failure_mode) for failure_mode in self.failure_modes]

        flat_results = [item for sublist in results for item in sublist]

        flat_results = flat_results+self.make_measurements_for_healthy({}) # Add the healthy data to the result

        return flat_results

class ClassificationDataset(object):
    def __init__(self, samples_per_class,failure_modes=["ball","inner","outer"],quick_iter=False, parallel_evaluate=False):
        self.failure_modes = failure_modes
        self.samples_per_class =samples_per_class
        self.parallel_evaluate = parallel_evaluate
        self.simulation_properties = get_simulation_properties(quick_iter=quick_iter)
        self.simulation_properties.update({"n_measurements":1}) # Only take one measurement per operating condition / severity

    def make_measurements_for_condition(self, properties_to_modify):
        """
        Modify the default parameters of the simulation as defined in simulation_properties.yml and create a
        measurement from a bearing measurement object

        Parameters
        ----------
        properties_to_modify: A dictionary of parameters that should be different from the default definition.

        Returns
        -------

        """

        # Make a copy of the default simulation properties
        modified_simulation_properties = self.simulation_properties.copy()

        # Update the simulation properties with the properties that are of interest for the current simulation
        modified_simulation_properties.update(properties_to_modify)

        # Define the measurement
        measurement_obj = BearingData(**modified_simulation_properties)
        meas = measurement_obj.get_measurements()  # Get the time_domain measurements

        # Create a dictionary with different flavours of the same pypm as well as meta pypm

        sampling_frequency = modified_simulation_properties["sampling_frequency"]

        meta_data = {"simulation_governing_parameters":modified_simulation_properties,
                     "sampling_frequency":sampling_frequency}


        meta_data.update({key:val for key,val in measurement_obj.derived_meta_data.items()}) # Add for example the mean rotation frequency

        if modified_simulation_properties["fault_severity"] == 0: # Healthy data has no mode
            meta_data["simulation_governing_parameters"].update({"fault_type":None})

        meas_dict = {"time_series": list(meas.flatten()),
                     "meta_data": meta_data}



        return meas_dict

    def make_measurements_for_different_failure_mode(self, properties_to_modify=None):
        """ For different failure modes, compute many samples for each severity"""

        if properties_to_modify is None:
            properties_to_modify = {}

        def process(failure_mode,severity):
            properties_to_modify["fault_type"] = failure_mode
            properties_to_modify["fault_severity"] = severity
            meas_at_mode = [self.make_measurements_for_condition(properties_to_modify) for _ in range(self.samples_per_class)]
            return meas_at_mode

        if self.parallel_evaluate:
            from joblib import Parallel, delayed

            failure_modes = self.failure_modes + [self.failure_modes[0]] # Add the final mode that will represent healthy case (severity = 0)
            severities = list(np.ones_like(self.failure_modes,dtype=float)) + [0] # The final severity with represent healthy
            results = Parallel(n_jobs=4)(delayed(process)(failure_mode,severity) for failure_mode,severity in zip(failure_modes,severities)) # TODO: njobs independent specify

            # results = [process(failure_mode) for failure_mode in self.failure_modes]

        flat_results = [item for sublist in results for item in sublist]


        return flat_results



def main():
    # o = PyBearingDataset(n_severities=3, failure_modes=["ball", "inner", "outer"], quick_iter=True) # TODO: Drive these parameters with governing yaml file
    # return o.make_measurements_for_different_failure_mode()

    o = LinearSeverityIncreaseDataset(n_test_samples=10,n_healthy_samples=10, failure_modes=["ball", "inner", "outer"], quick_iter=False,parallel_evaluate=True) # TODO: Drive these parameters with governing yaml file
    return o.make_measurements_for_different_failure_mode()



if __name__ == "__main__":
    r = main()
    # print(r)

# For classification dataset
# cd = ClassificationDataset(samples_per_class=2, failure_modes=["ball", "inner", "outer"],quick_iter=False,parallel_evaluate=True)
# r = cd.make_measurements_for_different_failure_mode()
# for i in r:
#     print(i["meta_data"])