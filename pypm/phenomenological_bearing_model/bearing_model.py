import numpy as np
import scipy.integrate as integrate
import scipy
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from pypm.utils.search import find_nearest_index


# TODO: Potentially making error at the edges when convolving (need to include additional pulses than requested by
#  the user to fix issue)
# This technicality could be largely resolved considering that the first impulse is at a radom starting position.

class Bearing(object):
    """
    Define fault characteristics of bearings based on bearing parameters.
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

        # Bearing parameters as class attributes
        self.d = d
        self.D = D
        self.contact_angle = contact_angle
        self.n_ball = n_ball

        # Compute derived parameters
        self.geometry_parameters = {"inner": self.get_inner_race_parameter,
                                    "outer": self.get_outer_race_parameter,
                                    "ball": self.get_ball_parameter}

    def get_inner_race_parameter(self):
        """
        Fault characteristic for inner race fault
        """
        return self.n_ball / 2 * (1 + self.d / self.D * np.cos(self.contact_angle))

    def get_outer_race_parameter(self):
        """
        Fault characteristic for outer race fault
        """
        return self.n_ball / 2 * (1 - self.d / self.D * np.cos(self.contact_angle))

    def get_ball_parameter(self):
        """
        Fault characteristic for ball fault
        """
        return self.D / (2 * self.d) * (1 - (self.d / self.D * np.cos(self.contact_angle)) ** 2)

    def get_geometry_parameter(self, fault_type):
        """
        Compute the geometry parameter for a given fault type
        """
        return self.geometry_parameters[fault_type]()

    def get_expected_fault_frequency(self,fault_type,rotation_frequency):
        """

        :param fault_type:
        :param rotation_frequency:  In Hz
        :return:
        """
        geometry_parameter = self.get_geometry_parameter(fault_type)
        average_fault_freq = rotation_frequency * geometry_parameter# / (2 * np.pi)

        return average_fault_freq

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
    """
    Take note that this makes the assumption that the frequency at which the speed profile is defined is the same as
    the master sampling frequency. This would lead to accurate integration to find angles, but is not strictly necessary.
    """

    def __init__(self, speed_profile_type, **kwargs):
        # Values for these attributes are set in the measurement class. Initialised here as None
        self.time = None
        self.n_master_samples = None
        self.n_measurements = None

        super().__init__(**kwargs)

        self.speed_profile_type = speed_profile_type

    def set_angles(self):
        """
        Compute the angles at a given master sample rate.
        Do this for each of the measurements.

        Returns
        -------

        """

        self.angles = self.get_angle_as_function_of_time()
        self.total_angle_traversed = self.get_total_angle_traversed()

    def get_rotation_angular_velocity_as_function_of_time(self):
        """
        Define the rotation rate as a function samples
        """

        profiles = {"constant": self.constant,
                    "sine": self.sine}
        return profiles[self.speed_profile_type]()  # Compute the appropriate speed profile

    # Further profiles can be defined here
    def constant(self):
        constant_speed = 500 * 2 * np.pi / 60  # 500 RPM in rad/s
        return np.ones((self.n_measurements, self.n_master_samples)) * constant_speed

    def sine(self):
        f = 2
        mean_speed = 500 * 2 * np.pi / 60  # 100 RPM in rad/s | revs/min * rads/rev * 2pi rads/rev * min/60s
        profile = mean_speed + np.sin(self.time * 2 * np.pi * f) * mean_speed * 0.5
        return np.outer(np.ones(self.n_measurements),
                        profile)  # Currently asigning same speed profile for each measurement

    def get_angle_as_function_of_time(self):
        """Integrate the speed profile to get the angle traversed as a function of time"""
        speed_profile = self.get_rotation_angular_velocity_as_function_of_time()

        angle = integrate.cumtrapz(y=speed_profile, x=self.time, axis=1, initial=0)  # Assume angle starts at 0

        return angle  # + initial_angle_for_each_measurement # Initial angle is added to all indexes for each row separately

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

        self.transient_time = np.linspace(0, self.transient_duration,
                                          int(self.transient_duration * self.master_sample_frequency))

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
        sdof_response = np.exp(-self.zeta * self.omegan * self.transient_time) * np.sin(
            self.omegad * self.transient_time)  # displacement
        sdof_response = sdof_response / np.max(sdof_response)
        return self.fault_severity * sdof_response
        # return np.ones(np.shape(sdof_reponse))


class Impulse():  # TODO: Possibly inherit from Bearing (or Gearbox) for that matter
    # TODO: Fix the init
    def __init__(self, slip_variance_factor, fault_type, randomized_starting_angle, **kwargs):
        super().__init__(**kwargs)

        self.variance_factor_angular_distance_between_impulses = slip_variance_factor
        self.fault_type = fault_type  # Fault type ie. "ball", "outer", "inner"

        self.randomized_starting_angle = randomized_starting_angle
        return

    def get_angular_distances_at_which_impulses_occur(self,
                                                      expected_number_of_impulses_during_measurement,
                                                      average_angular_distance_between_impulses):
        # Generate 5% than maximum more to account for  possibility of ending up with a shorter signal
        n_impulse_periods_to_generate = int(np.ceil(np.max(expected_number_of_impulses_during_measurement) * 1.05))
        # if std(expected_number_of_impulses) >threash : probably inefficient ie. generating too much pypm that will be discarded

        # Create array of the mean and variance for angular distance between impulses
        ones = np.ones((self.n_measurements, n_impulse_periods_to_generate))
        mean = ones * average_angular_distance_between_impulses

        # Generate random pypm of the distances traveled
        variance = mean * self.variance_factor_angular_distance_between_impulses
        distance_traveled_between_impulses = np.random.normal(mean, scale=np.sqrt(variance))

        # Add the distances between impulses together to find the distances from the start where impulses occur.
        cumulative_impulse_distance = np.cumsum(distance_traveled_between_impulses, axis=1)

        # use_random_starting_impulse = False
        # if use_random_starting_impulse:
        if self.randomized_starting_angle:
            random_starting_impulse_angle = np.random.uniform(0, average_angular_distance_between_impulses,
                                                              (self.n_measurements, 1))
        else:
            random_starting_impulse_angle = np.zeros((self.n_measurements, 1))

        # np.zeros((self.n_measurements, 1))
        angular_distances_at_which_impulses_occur = np.hstack(
            [random_starting_impulse_angle,
             cumulative_impulse_distance + random_starting_impulse_angle])  # An impulse occurs immediately
        # TODO: There are some stochastic issues when using large numbers of samples per dataset. Need to resolve this by starting with "negative time"
        return angular_distances_at_which_impulses_occur

    def get_indexes_at_which_impulse_occur_for_single_measurement(self, angles_for_measurement,
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
    def __init__(self, modulation_amplitude, angle_based_modulation_frequency, **kwargs):
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
        return 0.5 * (1 + np.cos(self.angle_based_modulation_frequency * 2 * np.pi * angles_for_all_measurements))

    def modulate_impulses_per_sample(self, impulses, angles_for_all_measurement):
        return impulses * self.get_normalized_modulation_signal(angles_for_all_measurement) * self.modulation_amplitude


class Measurement(Bearing, Impulse, SdofSys, SpeedProfile, Modulate):  # , Impulse):
    """
    Class used to define the properties of the dataset that will be simulated. Used to manage the overall pypm generation
    """

    def __init__(self,
                 n_measurements,
                 t_duration,
                 sampling_frequency,
                 measurement_noise_standard_deviation,
                 transient_amplitude_standard_deviation,
                 **kwargs):
        """

        Parameters
        ----------
        n_measurements: Number of measurements (Computation of measurements are mostly vectorized)
        measurement_duration: Duration of measurement in seconds [s].
        sampling_frequency Sampling frequency [Hz]
        """

        # Noise
        self.measurement_noise_standard_deviation = measurement_noise_standard_deviation
        self.transient_amplitude_standard_deviation = transient_amplitude_standard_deviation

        # Sampling properties
        self.sampling_frequency = sampling_frequency
        self.master_sample_frequency = int(
            self.sampling_frequency * 2)  # represents continuous time, the sampling frequency at which computations are performed.

        # Initialize the base classes
        super(Measurement, self).__init__(**kwargs)
        # Bearing.__init__(self,**kwargs) # Alternative is to initialize the base classes seperately since they have some interdependencies

        # Measurement attributes
        self.n_measurements = n_measurements
        self.t_duration = t_duration

        # Compute derived attributes
        self.n_measurement_samples = self.t_duration * self.sampling_frequency  # Total number of samples over the measurment interval

        self.n_master_samples = int(
            self.t_duration * self.master_sample_frequency)  # Total number of samples of the master samples "continuous time".

        self.time = np.linspace(0, self.t_duration,
                                self.n_master_samples)  # Time vector based on master sample rate.

        self.measured_time = self.time[
                             ::2]  # The measured time is sampled at half the simulation time (continuous time).

        # Set some derived parameters as meta data
        rotation_angular_velocity = self.get_rotation_angular_velocity_as_function_of_time() # rad/s
        mean_rotation_angular_velocity = np.mean(rotation_angular_velocity) # rad/s
        expected_fault_frequencies = {fault_type:self.get_expected_fault_frequency(fault_type, mean_rotation_angular_velocity/(2*np.pi))
                                      for fault_type in ["ball", "outer", "inner"]}

        self.derived_meta_data =  {
            "mean_rotation_frequency":np.mean(rotation_angular_velocity/(2*np.pi)), # cycle/second , Hz
            "t_duration":self.t_duration,
            "expected_fault_frequencies":expected_fault_frequencies
        }


    def add_measurement_noise(self, array):
        return np.random.normal(array, self.measurement_noise_standard_deviation)

    def get_measurements(self):
        """
        Compute the measurement set for the given parameters

        Returns
        -------
        Measurements array (number_of_measurements,number_of_samples)
        """

        # TODO: Rework some of the code below into the impulse class for clarity

        self.set_angles()  # Compute the angles from the speed profile
        average_angular_distance_between_impulses = self.get_angular_distance_between_impulses(self.fault_type)
        expected_number_of_impulses_during_measurement = self.total_angle_traversed / average_angular_distance_between_impulses

        angular_distances_at_which_impulses_occur = self.get_angular_distances_at_which_impulses_occur(
            expected_number_of_impulses_during_measurement,
            average_angular_distance_between_impulses)

        indexes = self.indexes_which_impulse_occur_for_all_measurement(angular_distances_at_which_impulses_occur)

        # If the fault type involves modulation, do modulation
        if self.fault_type == "inner":
            modulation_signal = self.modulate_impulses_per_sample(indexes, self.angles)
            indexes = modulation_signal

        # If there is a stochastic component to the amplitude of the transients in the phenomenological model

        # All indexes that are zero will remain zero,
        # those that are 1 are modified in magnitude
        # Standard deviation of 0.1 means that almost all of the data (2 standard deviations)
        # Will have an amplitude less than 20% different than otherwise
        indexes = indexes * (
                    1 - np.random.normal(np.zeros(indexes.shape), scale=self.transient_amplitude_standard_deviation))

        # Get the transient response for the SDOF system
        transient = self.get_transient_response()

        # Convolve the transient with the impulses
        convolved = scipy.signal.convolve2d(indexes, transient.reshape(1, -1), mode="same")
        # # convolved2 = scipy.signal.convolve(indexes_at_which_impulses_occur, transient.reshape(1,-1), mode="same")

        # Go from "continuous" time to measured time
        # measured = scipy.signal.decimate(convolved, 2, axis=1, ftype="iir") # Subsample from the master sample rate to the actual sample rate
        measured = convolved[:, ::2]  # Subsampling (get every second sample)

        # Add measurement noise
        measured = self.add_measurement_noise(measured)

        return measured


class BearingData(Measurement):
    def __init__(self, **kwargs):
        self.check_input_parameters(**kwargs)

        # Initialize the base classes
        super(BearingData, self).__init__(**kwargs)

    def check_input_parameters(self, **kwargs):
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
                         'measurement_noise_standard_deviation',
                         'transient_amplitude_standard_deviation',
                         'randomized_starting_angle',
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

        complement = list(set(keys_available) - set(keys_required))
        if len(complement) > 0:
            raise ValueError("To too many parameters specified. Spelling?" + str(complement))
