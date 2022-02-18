import numpy as np
from pypm.phenomenological_bearing_model.bearing_model import Measurement
from pypm.utils.reading_and_writing import get_simulation_properties


class PyBearingDataset(object):
    def __init__(self, n_severities, failure_modes, quick_iter=False):
        self.simulation_properties = get_simulation_properties(quick_iter=quick_iter)
        self.n_severities = n_severities
        self.failure_modes = failure_modes

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
        measurement_obj = Measurement(**modified_simulation_properties)
        meas = measurement_obj.get_measurements()  # Get the time_domain measurements

        # Create a dictionary with different flavours of the same pypm as well as meta pypm

        # Add derived meta-pypm to the meta pypm for later use
        modified_simulation_properties.update(
            measurement_obj.meta_data)

        meas_dict = {"time_domain": meas,
                     "meta_data": modified_simulation_properties}

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
        failure_mode_dict = {}

        from joblib import Parallel, delayed
        def process(failure_mode):
            properties_to_modify["fault_type"] = failure_mode
            return {failure_mode:self.make_measurements_for_different_severity(properties_to_modify)}


        results = Parallel(n_jobs=3)(delayed(process)(failure_mode) for failure_mode in self.failure_modes)

        failure_mode_dict = {key:val for x in results for key,val in x.items()}
        # # print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        # for result in results:
        #     failure_mode_dict.update(result)

        # # Include the pypm for each of the failure modes
        # for failure_mode in self.failure_modes:
        #     properties_to_modify["fault_type"] = failure_mode
        #     failure_mode_dict[failure_mode] = self.make_measurements_for_different_severity(properties_to_modify)

        return failure_mode_dict




def main():
    o = PyBearingDataset(n_severities=3, failure_modes=["ball", "inner", "outer"], quick_iter=True) # TODO: Drive these parameters with governing yaml file
    return o.make_measurements_for_different_failure_mode()



if __name__ == "__main__":
    main()