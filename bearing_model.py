import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from src.utils.search import find_nearest_index
import yaml

# TODO: Potentially making error at the edges when convolving (need to include additional pulses than requested by the user to adress this issue)
# TODO: Todo need to add amplitude modulation for the inner race fault.
# TODO: Let a global class inherit from a measurement object


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

        print("bearing init ran")

    def get_inner_race_parameter(self):
        """
        Fault characteristic for inner race fault
        """
        return 1 / 2 * (1 + self.d / self.D * np.cos(self.contact_angle))

    def get_outer_race_parameter(self):
        """
        Fault characteristic for outer race fault
        """
        return 1 / 2 * (1 - self.d / self.D * np.cos(self.contact_angle))

    def get_ball_parameter(self):
        """
        Fault characteristic for ball fault
        """
        return 1 / (2 * self.n_ball) * (1 - (self.d / self.D * np.cos(self.contact_angle)) ** 2) / (self.d / self.D)

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

        geometry_parameter = self.get_geometry_parameter(fault_type)
        impulses_per_revolution = geometry_parameter * self.n_ball

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

        print("speed profile init ran")

    def set_angles(self):
        self.angles = self.get_angle_as_function_of_time()
        self.total_angle_traversed = self.get_total_angle_traversed()

    def get_rotation_frequency_as_function_of_time(self):
        """Define the rotation rate as a function samples"""
        profiles = {"constant": self.constant,
                    "sine": self.sine}
        return profiles[self.speed_profile_type]()  # Compute the appropriate speed profile

    def constant(self):
        constant_speed = 100 * 2 * np.pi / 60  # 100 RPM in rad/s
        return np.ones((self.n_measurements, self.n_master_samples)) * constant_speed

    def sine(self):
        f = 5
        mean_speed = 100 * 2 * np.pi / 60  # 100 RPM in rad/s
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
        super().__init__(**kwargs)
        m = k / (2 * np.pi * fn) ** 2
        F = fault_severity
        self.zeta = zeta

        self.A = F / m
        self.omegan = 2 * np.pi * fn
        self.omegad = self.omegan * np.sqrt(1 - zeta ** 2)

        percentage = 0.01  # Percentage of the original amplitude to decay to
        self.transient_duration = np.log(1 / percentage) / (self.zeta * self.omegan)

        self.t_range = np.linspace(0, self.transient_duration, int(self.transient_duration * self.master_sample_frequency))

    def get_transient_response(self):
        """
        Compute the transient response of signle degree of freedom system subjected to step input.

        Returns
        -------

        """
        xt = self.A / self.omegad * np.exp(-self.zeta * self.omegan * self.t_range) * np.sin(
            self.omegad * self.t_range)  # displacement
        xd = np.hstack([[0], np.diff(xt) * self.master_sample_frequency])  # velocity
        sdof_reponse = np.hstack(
            [[0], np.diff(xd) * self.master_sample_frequency])  # acceleration #TODO: Use the analytical formula and not double integration
        return sdof_reponse


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


class Measurement(Bearing,Impulse, SdofSys,SpeedProfile):#, Impulse):
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

        self.n_master_samples = self.t_duration * self.master_sample_frequency  # Total number of samples of the master samples "continuous time".
        self.time = np.linspace(0, self.t_duration,
                                self.n_master_samples)  # Time vector based on master sample rate.

    def check_input_parameters(self,**kwargs):
        """
        Check which of the entries of the provided dictionary might be missing.
        
        Returns
        -------
        """
        required_parameters = ['fault_severity',
                               'fault_type',
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

        print(kwargs.keys())


    def get_measurements(self):
        """
        Compute the measurement set for the given parameters

        Returns
        -------
        Measurements array (number_of_measurements,number_of_samples)
        """

        self.set_angles()  # Compute the angles from the speed profile
        average_angular_distance_between_impulses = self.get_angular_distance_between_impulses("ball")
        expected_number_of_impulses_during_measurement = self.total_angle_traversed / average_angular_distance_between_impulses

        angular_distances_at_which_impulses_occur = self.get_angular_distances_at_which_impulses_occur(
            expected_number_of_impulses_during_measurement,
            average_angular_distance_between_impulses)

        indexes = self.indexes_which_impulse_occur_for_all_measurement(angular_distances_at_which_impulses_occur)

        # Get the transient response for the SDOF system
        transient = self.get_transient_response()

        # Convolve the transient with the impulses

        convolved = scipy.signal.convolve2d(indexes, transient.reshape(1, -1), mode="same")
        # # convolved2 = scipy.signal.convolve(indexes_at_which_impulses_occur, transient.reshape(1,-1), mode="same")


        # # measured = scipy.signal.decimate(measurement, 2, axis=1, ftype="fir") # Subsample from the master sample rate to the actual sample rate
                                                                                  # Low pass filter to prevent ani-aliasing
        return convolved

