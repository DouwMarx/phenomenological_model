import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from src.utils.search import find_nearest_index

# Generation of a simulated signal for localized fault in rolling element bearing

# TODO: Add this to a yaml file
d = 8.4  # bearing roller diameter [mm]
D = 71.5  # pitch circle diameter [mm]
n_ball = 16  # number of rolling elements
contactAngle = 15.7 * np.pi / 180  # contact angle
faultType = 3  # fault type selection: inner, outer, ball [string]
fc = 3  # row vector containing the carrier component of the speed
fm = 3  # row vector containing the modulation frequency
fd = 3  # row vector containing the frequency deviation
N = 3  # number of points per revolution
variance_factor_angular_distance_between_impulses = 0.0011  # variance for the generation of the random contribution (ex. 0.04)
fs = 3  # sample frequency of the time vector
k = 2e13  # SDOF stiffness
zita = 0.02  # SDOF damping
fn = 4230  # SDOF natural freq [Hz]
Lsdof = 3  # length of the in number of points of the SDOF response
SNR_dB = 3  # signal to noise ratio [dB]
qAmpMod = 1  # amplitude modulation due to the load (ex. 0.3)


# TODO: Possibly return these metrics as validation
# meanDeltaT = theoretical mean of the inter-arrival times
# varDeltaT = theoretical variance of the inter-arrival times
# menDeltaTimpOver = real mean of the inter-arrival times
# varDeltaTimpOver = real variance of the inter-arrival times
# errorDeltaTimp = generated error in the inter-arrival times

class Bearing(object):
    """
    Class for defining fault characteristics of bearings based on bearing parameters.
    """

    def __init__(self, d, D, contact_angle, n_ball,**kwargs):
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


class SpeedProfile(object):
    # TODO: This makes the assumption that the frequency at which the speed profile is defined is the same as the sampling frequency which is not strictly required
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
        # integrate_a = np.cumsum(sp[:,0:-1] + 0.5*np.diff(sp,axis=1),axis=1)/m.master_frequency can remake integration in numpy only if required
        speed_profile = self.get_rotation_frequency_as_function_of_time()
        angle = integrate.cumtrapz(y=speed_profile, x=self.time, axis=1, initial=0)  # Assume angle starts at 0
        return angle

    def get_total_angle_traversed(self):
        return self.angles[:,
               -1]  # Total angular distance traveled for each measurement at the end of the measurement interval


class SdofSys(object):
    # %% Acceleration of a SDOF system
    # % [sdofRespTime] = sdofResponse(fs,k,zita,fn,Lsdof)
    # %
    # % Input:
    # % fs = sample frequency [Hz]
    # % k = spring stiffness [N/m]
    # % zita = damping coefficient
    # % fn = Natural frequency [Hz]
    # % Lsdof = desired signal length [points]
    # %
    # % Output:
    # % sdofRespTime = acceleration (row vector)
    # def __init__(self, fs=1000, k=100, zeta=0.1, fn=100, F=1):
    def __init__(self, fs, k, zeta, fn, F,**kwargs):
        super().__init__(**kwargs)
        self.m = k / (2 * np.pi * fn) ** 2
        self.fs = fs

        self.F = 1
        self.zeta = zeta

        self.A = F / self.m
        self.omegan = 2 * np.pi * fn
        self.omegad = self.omegan * np.sqrt(1 - zeta ** 2)

        # e^0 / e^(-zeta*omegan*t_end) < 0.01 # Need to decay to 1% of original amplitude
        # 1 < percentage*e^(-zeta*omegan*t_end)
        # log(1/percentage)/(-zeta*omegan) < t_end
        # t_end > log(1/percentage)/(-zeta*omegan)

        percentage = 0.01
        self.transient_duration = np.log(1 / percentage) / (self.zeta * self.omegan)

        self.t_range = np.linspace(0, self.transient_duration, int(self.transient_duration * fs))

        print("sdof sys profile init ran")

    def get_transient_response(self):
        # SDOF response
        xt = self.A / self.omegad * np.exp(-self.zeta * self.omegan * self.t_range) * np.sin(
            self.omegad * self.t_range)  # displacement
        xd = np.hstack([[0], np.diff(xt) * fs])  # velocity
        sdof_reponse = np.hstack([[0], np.diff(xd) * fs])  # velocity
        return sdof_reponse
        # return 5000*xt


class Impulse(object): # TODO: Possibly inherit from Bearing (or Gearbox) for that matter
    # TODO: Fix the init
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print("impulse init ran")
        return

    def get_angular_distances_at_which_impulses_occur(self,
                                                      expected_number_of_impulses_during_measurement,
                                                      average_angular_distance_between_impulses,
                                                      variace_factor_angular_distance_between_impulses):
        # Generate 5% than maximum more to account for  possibility of ending up with a shorter signal
        n_impulse_periods_to_generate = int(np.ceil(np.max(expected_number_of_impulses_during_measurement) * 1.05))
        # if std(expected_number_of_impulses) >threash : probably inefficient ie. generating too much data that will be discarded

        # Create array of the mean and variance for angular distance between impulses
        ones = np.ones((m.n_measurements, n_impulse_periods_to_generate))
        mean = ones * average_angular_distance_between_impulses

        # Generate random data of the distances traveled
        variance = mean * variace_factor_angular_distance_between_impulses
        distance_traveled_between_impulses = np.random.normal(mean, scale=np.sqrt(variance))

        # Add the distances between impulses together to find the distances from the start where impulses occur.
        cumulative_impulse_distance = np.cumsum(distance_traveled_between_impulses, axis=1)
        angular_distances_at_which_impulses_occur = np.hstack(
            [np.zeros((m.n_measurements, 1)), cumulative_impulse_distance])  # An impulse occurs immediately
        # TODO: Add random variation in when the fist impulse occurs
        return angular_distances_at_which_impulses_occur

    @staticmethod
    def get_indexes_at_which_impulse_occur_for_single_measurement(angles_for_measurement,impulse_distances_in_measurement_interval):
        # Find the times corresponding to a given angle
        times_corresponding_to_angle = interp1d(angles_for_measurement, m.time)(impulse_distances_in_measurement_interval)

        # Find the indexes that is closest to the times at which the impulses occur
        indexes = find_nearest_index(m.time, times_corresponding_to_angle)
        return indexes

    @staticmethod
    def get_impulses_distances_in_measurement_interval(impulse_distances_for_measurement,total_angle_traversed_for_measurement):
        # Discard the impulses that fall outside of the ranges of total angles traversed
        return impulse_distances_for_measurement[impulse_distances_for_measurement < total_angle_traversed_for_measurement]

    def indexes_which_impulse_occur_for_all_measurement(self,angular_distances_at_which_impulses_occur):
        indexes_at_which_impulses_occur = np.zeros((m.n_measurements,
                                                    m.n_master_samples))  # Initialize empty array that show (master) samples where impulses will occur
        for measurement in range(m.n_measurements):
            # For each separate measurement
            angles_for_measurement = self.angles[measurement, :]
            impulse_distances_for_measurement = angular_distances_at_which_impulses_occur[measurement, :]
            total_angle_traversed_for_measurement = self.total_angle_traversed[measurement]

            impulse_distances_in_measurement_interval = self.get_impulses_distances_in_measurement_interval(impulse_distances_for_measurement,total_angle_traversed_for_measurement)

            indexes = self.get_indexes_at_which_impulse_occur_for_single_measurement(angles_for_measurement,impulse_distances_in_measurement_interval)

            # Set the indexes where the impulses occur to 1 (Otherwise zero)
            indexes_at_which_impulses_occur[measurement, indexes] = 1

        return indexes_at_which_impulses_occur


class Measurement(Bearing,SpeedProfile, SdofSys, Impulse):
    """
    Class used to define the properties of the dataset that will be simulated. Used to manage the overall data generation
    """

    def __init__(self, n_measurements=5, measurement_duration=1, sampling_frequency=40000):
        """

        Parameters
        ----------
        n_measurements: Number of measurements (Computation of measurements are mostly vectorized)
        measurement_duration: Duration of measurement in seconds [s].
        sampling_frequency Sampling frequency [Hz]
        """
        # Initialize the base classes
        super(Measurement, self).__init__(speed_profile_type="constant",
                                          fs=1000,
                                          k=1000,
                                          zeta=0.1,
                                          fn=100,
                                          F=1,
                                          d=0.02,
                                          D = 0.1,
                                          contact_angle =3,
                                          n_ball = 10)

        # Measurement attributes
        self.n_measurements = n_measurements
        self.measurement_duration = measurement_duration
        self.sampling_frequency = sampling_frequency

        # Operating condition attributes

        # Compute derived attributes
        self.master_frequency = int(
            self.sampling_frequency * 2)  # represents continuous time, the sampling frequency at which computations are performed.

        self.n_measurement_samples = self.measurement_duration * self.sampling_frequency  # Total number of samples over the measurment interval

        self.n_master_samples = self.measurement_duration * self.master_frequency  # Total number of samples of the master samples "continuous time".
        self.time = np.linspace(0, self.measurement_duration,
                                self.n_master_samples)  # Time vector based on master sample rate.

    def get_measurements(self):
        self.set_angles() # Compute the angles from the speed profile
        average_angular_distance_between_impulses = self.get_angular_distance_between_impulses("ball")
        expected_number_of_impulses_during_measurement = self.total_angle_traversed / average_angular_distance_between_impulses

        angular_distances_at_which_impulses_occur = self.get_angular_distances_at_which_impulses_occur(
            expected_number_of_impulses_during_measurement,
            average_angular_distance_between_impulses,
            variance_factor_angular_distance_between_impulses)

        indexes = self.indexes_which_impulse_occur_for_all_measurement(angular_distances_at_which_impulses_occur)

        # Get the transient response for the SDOF system
        transient = self.get_transient_response()

        # Convolve the transient with the impulses

        convolved = scipy.signal.convolve2d(indexes, transient.reshape(1, -1), mode="same")
        # # convolved2 = scipy.signal.convolve(indexes_at_which_impulses_occur, transient.reshape(1,-1), mode="same")
        # # These two options need to be looped over, can test performance
        # # plt.plot(np.convolve(transient,indexes_at_which_impulses_occur[0]))
        # # plt.plot(fftconvolve(transient,indexes_at_which_impulses_occur[0]))

        return convolved



# TODO: Need to have Measurement(bearing_properties,measurement_properties)
# TODO: change the starting angle at which the fist fault occurs

m = Measurement()
measurement = m.get_measurements()

# TODO: Potentially making error at the edges when convolving (need to include additional pulses than requested by the user to adress this issue)

# measured = scipy.signal.decimate(measurement, 2, axis=1, ftype="fir")

plt.figure()
# plt.plot(indexes_at_which_impulses_occur[0]*np.max(convolved[0]))
plt.plot(measurement[0])

# TODO: Todo need to add amplitude modulation for the inner race fault.

