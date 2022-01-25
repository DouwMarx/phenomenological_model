import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from src.utils.search import find_nearest_index
import pickle
import yaml

# TODO: Todo need to add amplitude modulation for the inner race fault.

# TODO: Potentially making error at the edges when convolving (need to include additional pulses than requested by the user to adress this issue)

# TODO: Let a global class inherit from a measurement object, perhaps call it Dataset


class Bearing():
    """
    Class for defining fault characteristics of bearings based on bearing parameters.
    """

    def __init__(self, d, D, contact_angle, n_ball, **kwargs):
        """

        Parameters
        ----------
        d: Bearing roller diameter [mm]
        D: Pitch circle diameter [mm]
        contact_angle:  Contact angle [rad]
        n_ball: Number of rolling elements
        """
        super().__init__(**kwargs)

        # Assign bearing parameters as class attributes
        self.d = d
        self.D = D
        self.contact_angle = contact_angle
        self.n_ball = n_ball

        # Compute derived paramters
        self.geometry_parameters = {"inner": self.get_inner_race_parameter,
                                    "outer": self.get_outer_race_parameter,
                                    "ball": self.get_ball_parameter}
        # print("bearing init ran")


    # TODO: This definition is strange
    # Ideally it would be factor = f_impulse/f_rotation

    # TODO:Double check the fault frequencies. There was mistake?
    def get_inner_race_parameter(self):
        """
        Fault characteristic for inner race fault
        """
        return self.n_ball / 2 * (1 + self.d / self.D * np.cos(self.contact_angle))

    def get_outer_race_parameter(self):
        """
        Fault characteristic for outer race fault
        """
        return  self.n_ball / 2 * (1 - self.d / self.D * np.cos(self.contact_angle))

    def get_ball_parameter(self):
        """
        Fault characteristic for ball fault
        """
        return  self.D / (2*self.d) * (1 - (self.d / self.D * np.cos(self.contact_angle)) ** 2)

    def get_geometry_parameter(self, fault_type):
        """
        Compute the geometry parameter for a given fault type
        """
        return self.geometry_parameters[fault_type]()

    def get_angular_distance_between_impulses(self, fault_type):
        """
        Compute the expected angular distance that the bearing would rotate between impulses due to a given fault mode.

        Parameters
        ----------
        fault_type: "inner","outer","ball"

        Returns
        average angular distance between impulses [radians].
        -------
        """

        impulses_per_revolution = self.get_geometry_parameter(fault_type)

        average_angular_distance_between_impulses = 2 * np.pi / impulses_per_revolution
        # (rads/rev)/(impulses/rev) = rads/impulse

        return average_angular_distance_between_impulses


class SpeedProfile():
    # TODO: This makes the assumption that the frequency at which the speed profile is defined is the same as the sampling frequency which is not strictly required
    # TODO: change the starting angle at which the fist fault occurs
    def __init__(self, speed_profile_type, **kwargs):
        # Values for these attributes are set in the measurement class
        self.time = None
        self.n_master_samples = None
        self.n_measurements = None

        super().__init__(**kwargs)

        self.speed_profile_type = speed_profile_type

    def set_angles(self):
        """
        Compute the angles at a given master sample.
        Do this for each of the measurements.
        Returns
        -------

        """
        self.angles = self.get_angle_as_function_of_time()
        self.total_angle_traversed = self.get_total_angle_traversed()

    def get_rotation_frequency_as_function_of_time(self):
        """Define the rotation rate as a function samples"""
        profiles = {"constant": self.constant,
                    "sine": self.sine}
        return profiles[self.speed_profile_type]()  # Compute the appropriate speed profile

    def constant(self):
        constant_speed = 500 * 2 * np.pi / 60  # 500 RPM in rad/s
        return np.ones((self.n_measurements, self.n_master_samples)) * constant_speed

    def sine(self):
        f = 5
        mean_speed = 100 * 2 * np.pi / 60  # 100 RPM in rad/s | revs/min * rads/rev * 2pi rads/rev * min/60s
        profile = mean_speed + np.sin(self.time * 2 * np.pi * f) * mean_speed * 0.9
        return np.outer(np.ones(self.n_measurements), profile)

    def get_angle_as_function_of_time(self):
        """Integrate the speed profile to get the angle traversed as a function of time"""
        # integrate_a = np.cumsum(sp[:,0:-1] + 0.5*np.diff(sp,axis=1),axis=1)/m.master_sample_frequency can remake integration in numpy only if required
        speed_profile = self.get_rotation_frequency_as_function_of_time()
        angle = integrate.cumtrapz(y=speed_profile, x=self.time, axis=1, initial=0)  # Assume angle starts at 0
        return angle

    def get_total_angle_traversed(self):
        return self.angles[:,
               -1]  # Total angular distance traveled for each measurement at the end of the measurement interval


class SdofSys():
    def __init__(self, k, zeta, fn, fault_severity, **kwargs):
        """
        Return the acceleration of a single degree of freedom dynamical system that with a step input.

        Parameters
        ----------
        sampling_frequency: sample frequency [Hz] #TODO: Delete this later: get from other properties
        k: spring stiffness [N/m]
        zeta: damping ratio
        fn: natural frequency [Hz]
        fault_severity: Step input magnitude [N]
        kwargs

        Derivation of required transient length
        --------------------------------------
        We would like the transient to decay to 1% of its original amplitude

        e^0 / e^(-zeta*omegan*t_end) < 0.01
        1 < percentage*e^(-zeta*omegan*t_end)
        log(1/percentage)/(-zeta*omegan) < t_end
        t_end > log(1/percentage)/(-zeta*omegan)

        """

        # TODO: make sure I use a different name than trange since the global time is something like time
        super().__init__(**kwargs)
        m = k / (2 * np.pi * fn) ** 2

        self.fault_severity = fault_severity
        F = fault_severity
        self.zeta = zeta

        self.A = F / m
        self.omegan = 2 * np.pi * fn
        self.omegad = self.omegan * np.sqrt(1 - zeta ** 2)

        percentage = 0.01  # Percentage of the original amplitude to decay to # TODO: Add to kwargs
        self.transient_duration = np.log(1 / percentage) / (self.zeta * self.omegan)

        self.transient_time= np.linspace(0, self.transient_duration, int(self.transient_duration * self.master_sample_frequency))

    def get_transient_response(self):
        """
        Compute the transient response of signle degree of freedom system subjected to step input.

        Returns
        -------

        """
        # xt = self.A / self.omegad * np.exp(-self.zeta * self.omegan * self.t_range) * np.sin(
        #     self.omegad * self.t_range)  # displacement
        # xd = np.hstack([[0], np.diff(xt) * self.master_sample_frequency])  # velocity
        # sdof_reponse = np.hstack(
        #     [[0], np.diff(xd) * self.master_sample_frequency])  # acceleration #TODO: Use the analytical formula and not double integration

        # sdof_reponse = -self.zeta * np.exp(-self.omegan * self.t_range * self.zeta) * np.sin(
        #     self.omegan * self.t_range * np.sqrt(np.abs(self.zeta ** 2 - 1))) / (
        #             2 * np.sqrt(np.abs(self.zeta ** 2 - 1))) + np.exp(-self.omegan * self.t_range * self.zeta) * np.cos(
        #     self.omegan * self.t_range * np.sqrt(np.abs(self.zeta ** 2 - 1))) + np.exp(
        #     -self.omegan * self.t_range * self.zeta) * np.sin(
        #     self.omegan * self.t_range * np.sqrt(np.abs(self.zeta ** 2 - 1))) * np.sqrt(np.abs(self.zeta ** 2 - 1)) / (
        #             2 * self.zeta)

        # TODO: Below is a temporary fix, Assumption that the acceleration starts from 0 and then increases, fault severity is max of transient
        sdof_response = np.exp(-self.zeta * self.omegan * self.transient_time) * np.sin(self.omegad * self.transient_time)  # displacement
        sdof_response = sdof_response/np.max(sdof_response)
        return self.fault_severity*sdof_response
        # return np.ones(np.shape(sdof_reponse))


class Impulse():  # TODO: Possibly inherit from Bearing (or Gearbox) for that matter
    # TODO: Fix the init
    def __init__(self, slip_variance_factor,fault_type, **kwargs):
        super().__init__(**kwargs)
        # super().__init__()

        self.variance_factor_angular_distance_between_impulses = slip_variance_factor
        self.fault_type =fault_type # Fault type ie. "ball", "outer", "inner"
        return

    def get_angular_distances_at_which_impulses_occur(self,
                                                      expected_number_of_impulses_during_measurement,
                                                      average_angular_distance_between_impulses):
        # Generate 5% than maximum more to account for  possibility of ending up with a shorter signal
        n_impulse_periods_to_generate = int(np.ceil(np.max(expected_number_of_impulses_during_measurement) * 1.05))
        # if std(expected_number_of_impulses) >threash : probably inefficient ie. generating too much data that will be discarded

        # Create array of the mean and variance for angular distance between impulses
        ones = np.ones((self.n_measurements, n_impulse_periods_to_generate))
        mean = ones * average_angular_distance_between_impulses

        # Generate random data of the distances traveled
        variance = mean * self.variance_factor_angular_distance_between_impulses
        distance_traveled_between_impulses = np.random.normal(mean, scale=np.sqrt(variance))

        # Add the distances between impulses together to find the distances from the start where impulses occur.
        cumulative_impulse_distance = np.cumsum(distance_traveled_between_impulses, axis=1)
        angular_distances_at_which_impulses_occur = np.hstack(
            [np.zeros((self.n_measurements, 1)), cumulative_impulse_distance])  # An impulse occurs immediately
        # TODO: Add random variation in when the fist impulse occurs
        return angular_distances_at_which_impulses_occur

    def get_indexes_at_which_impulse_occur_for_single_measurement(self,angles_for_measurement,
                                                                  impulse_distances_in_measurement_interval):
        # Find the times corresponding to a given angle
        times_corresponding_to_angle = interp1d(angles_for_measurement, self.time)(
            impulse_distances_in_measurement_interval)

        # Find the indexes that is closest to the times at which the impulses occur
        indexes = find_nearest_index(self.time, times_corresponding_to_angle)
        return indexes

    @staticmethod
    def get_impulses_distances_in_measurement_interval(impulse_distances_for_measurement,
                                                       total_angle_traversed_for_measurement):
        # Discard the impulses that fall outside of the ranges of total angles traversed
        return impulse_distances_for_measurement[
            impulse_distances_for_measurement < total_angle_traversed_for_measurement]

    def indexes_which_impulse_occur_for_all_measurement(self, angular_distances_at_which_impulses_occur):
        """
        Loop over all measurements to get the indexes for which impulses occur.
        Parameters
        ----------
        angular_distances_at_which_impulses_occur

        Returns
        -------

        """
        indexes_at_which_impulses_occur = np.zeros((self.n_measurements,
                                                    self.n_master_samples))  # Initialize empty array that show (master) samples where impulses will occur
        for measurement in range(self.n_measurements):
            # For each separate measurement
            angles_for_measurement = self.angles[measurement, :]
            impulse_distances_for_measurement = angular_distances_at_which_impulses_occur[measurement, :]
            total_angle_traversed_for_measurement = self.total_angle_traversed[measurement]

            impulse_distances_in_measurement_interval = self.get_impulses_distances_in_measurement_interval(
                impulse_distances_for_measurement, total_angle_traversed_for_measurement)

            indexes = self.get_indexes_at_which_impulse_occur_for_single_measurement(angles_for_measurement,
                                                                                     impulse_distances_in_measurement_interval)

            # Set the indexes where the impulses occur to 1 (Otherwise zero)
            indexes_at_which_impulses_occur[measurement, indexes] = 1

        return indexes_at_which_impulses_occur

class Modulate():
    def __init__(self, modulation_amplitude,angle_based_modulation_frequency,**kwargs):
        super().__init__(**kwargs)

        self.modulation_amplitude = modulation_amplitude
        self.angle_based_modulation_frequency = angle_based_modulation_frequency
        return

    def get_normalized_modulation_signal(self, angles_for_all_measurements):
        """
        Computes the angle dependent modulation signal scaled between 0 and 1
        This is computed for all measurements
        Parameters
        ----------
        angles_for_all_measurements

        Returns
        -------

        """
        return 0.5*(1+np.cos(self.angle_based_modulation_frequency*2*np.pi*angles_for_all_measurements))

    def modulate_impulses_per_sample(self,impulses,angles_for_all_measurement):
        return impulses*self.get_normalized_modulation_signal(angles_for_all_measurement)*self.modulation_amplitude

class Measurement(Bearing,Impulse, SdofSys,SpeedProfile,Modulate):#, Impulse):
    """
    Class used to define the properties of the dataset that will be simulated. Used to manage the overall data generation
    """

    # TODO: let paramets be pulled from flattened dict from yaml
    def __init__(self,
                 n_measurements,
                 t_duration,
                 sampling_frequency,
                 measurement_noise_variance,
                 **kwargs):
        """

        Parameters
        ----------
        n_measurements: Number of measurements (Computation of measurements are mostly vectorized)
        measurement_duration: Duration of measurement in seconds [s].
        sampling_frequency Sampling frequency [Hz]
        """

        # self.check_input_parameters(**kwargs) # TODO: need to run a check that gives the user information if the wrong arguments are provided

        # Sampling properties
        self.sampling_frequency = sampling_frequency
        self.master_sample_frequency = int(self.sampling_frequency * 2)  # represents continuous time, the sampling frequency at which computations are performed.

        # Initialize the base classes
        super(Measurement, self).__init__(**kwargs)
        #Bearing.__init__(self,**kwargs) # Alternative is to initialize the base classes seperately since they have some interdependencies

        # Measurement attributes
        self.n_measurements = n_measurements
        self.t_duration = t_duration

        # Operating condition attributes

        # Compute derived attributes

        self.n_measurement_samples = self.t_duration * self.sampling_frequency  # Total number of samples over the measurment interval

        self.n_master_samples = int(self.t_duration * self.master_sample_frequency)  # Total number of samples of the master samples "continuous time".
        self.time = np.linspace(0, self.t_duration,
                                self.n_master_samples)  # Time vector based on master sample rate.


        # Set some derived parameters as meta data
        self.meta_data = {"derived": {
            "geometry_factor": self.get_geometry_parameter(self.fault_type)
        }}

    def check_input_parameters(self,**kwargs):
        """
        Check which of the entries of the provided dictionary might be missing.
        
        Returns
        -------
        """
        keys_required = ['fault_severity',
                               'fault_type',
                               'modulation_amplitude',
                               'angle_based_modulation_frequency',
                               'slip_variance_factor',
                               'measurement_noise_variance',
                               'd',
                               'D',
                               'n_ball',
                               'contact_angle',
                               'sampling_frequency',
                               't_duration',
                               'n_measurements',
                               'k',
                               'zeta',
                               'fn',
                               'speed_profile_type']

        keys_available = (kwargs.keys())

        complement =list(set(keys_required) - set(keys_available))
        print(complement)

        if len(complement) > 0:
            raise ValueError("To few or too many parameters")


    def get_measurements(self):
        """
        Compute the measurement set for the given parameters

        Returns
        -------
        Measurements array (number_of_measurements,number_of_samples)
        """

        #TODO: Rework some of the code below into the impulse class for clarity

        self.set_angles()  # Compute the angles from the speed profile
        average_angular_distance_between_impulses = self.get_angular_distance_between_impulses(self.fault_type)
        expected_number_of_impulses_during_measurement = self.total_angle_traversed / average_angular_distance_between_impulses

        angular_distances_at_which_impulses_occur = self.get_angular_distances_at_which_impulses_occur(
            expected_number_of_impulses_during_measurement,
            average_angular_distance_between_impulses)

        indexes = self.indexes_which_impulse_occur_for_all_measurement(angular_distances_at_which_impulses_occur)


        if self.fault_type == "inner": # If the fault time ivolves modulation, do modulation

            modulation_signal = self.modulate_impulses_per_sample(indexes, self.angles)
            indexes = modulation_signal


        # Get the transient response for the SDOF system
        transient = self.get_transient_response()

        # Convolve the transient with the impulses

        convolved = scipy.signal.convolve2d(indexes, transient.reshape(1, -1), mode="same")
        # # convolved2 = scipy.signal.convolve(indexes_at_which_impulses_occur, transient.reshape(1,-1), mode="same")
        # return convolved

        # measured = scipy.signal.decimate(convolved, 2, axis=1, ftype="fir") # Subsample from the master sample rate to the actual sample rate
        # measured = scipy.signal.decimate(convolved, 2, axis=1, ftype="iir") # Subsample from the master sample rate to the actual sample rate
        measured = convolved[:,::2] # Subsampling (get every second sample)
        # measured = convolved

        return measured

                                                                                  # Low pass filter to prevent ani-aliasing

    # def generate_data_and_export(self):
    #     p
    #
