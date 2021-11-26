import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from src.utils.search import find_nearest_index

# Generation of a simulated signal for localized fault in rolling element bearing

d = 8.4  # bearing roller diameter [mm]
D = 71.5  # pitch circle diameter [mm]
n_ball = 16  # number of rolling elements
contactAngle = 15.7 * np.pi / 180  # contact angle
faultType = 3  # fault type selection: inner, outer, ball [string]
fr = 3  # row vector containing the rotation frequency profile
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


# Output:
# t = time signal [s]
# x = simulated bearing signal without noise
# xNoise = simulated bearing signal with noise
# frTime = speed profile in the time domain [Hz]
# meanDeltaT = theoretical mean of the inter-arrival times
# varDeltaT = theoretical variance of the inter-arrival times
# menDeltaTimpOver = real mean of the inter-arrival times
# varDeltaTimpOver = real variance of the inter-arrival times
# errorDeltaTimp = generated error in the inter-arrival times

class Bearing(object):
    def __init__(self, d, D, contact_angle, n_ball):
        self.d = d
        self.D = D
        self.contact_angle = contact_angle

        self.geometry_parameters = {"inner": self.get_inner_race_parameter,
                                    "outer": self.get_outer_race_parameter,
                                    "ball": self.get_ball_parameter}

        self.n_ball = n_ball

    def get_inner_race_parameter(self):
        """
        For inner race fault
        """
        return 1 / 2 * (1 + self.d / self.D * np.cos(self.contact_angle))

    def get_outer_race_parameter(self):
        """
        For outer race fault
        """
        return 1 / 2 * (1 - self.d / self.D * np.cos(self.contact_angle))

    def get_ball_parameter(self):
        """
        For ball fault
        """
        return 1 / (2 * self.n_ball) * (1 - (self.d / self.D * np.cos(self.contact_angle)) ** 2) / (self.d / self.D)

    def get_geometry_parameter(self, fault_type):
        """Compute the geometry parameter for a given fault type"""
        return self.geometry_parameters[fault_type]()

    def get_angular_distance_between_impulses(self, fault_type):
        geometry_parameter = self.get_geometry_parameter(fault_type)
        impulses_per_revolution = geometry_parameter * self.n_ball
        average_angular_distance_between_impulses = 2 * np.pi / impulses_per_revolution  # (rads/rev)/(impulses/rev) = rads/impulse
        return average_angular_distance_between_impulses


class Measurement():
    def __init__(self):
        # User set
        self.n_measurements = 5  # Number of different measurements
        # TODO: Should be able to specify either sample duration or number of samples (not number of revs)
        self.measurement_duration = 1  # Duration of a measurement [s]
        self.sampling_frequency = 40000  # Sampling frequency [Hz]

        # Derived
        self.master_frequency = int(
            self.sampling_frequency * 2)  # Represents continuous time, the sampling frequency at which computations are performed.
        self.n_measurement_samples = self.measurement_duration * self.sampling_frequency  # The total number of samples over the measurment interval
        self.n_master_samples = self.measurement_duration * self.master_frequency  # The total number of samples of the master samples

        self.time = np.linspace(0, self.measurement_duration, self.n_master_samples)


class SpeedProfile(Measurement):
    # TODO: This makes the assumption that the frequency at which the speed profile is defined is the same as the sampling frequency which is not strictly required
    def __init__(self, speed_profile_type="constant"):
        super().__init__()  # Initialize the measurement class
        self.speed_profile_type = speed_profile_type
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


class SdofSys():
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
    def __init__(self, fs, k, zeta, fn, F):
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

    def get_transient_response(self):
        # SDOF response
        xt = self.A / self.omegad * np.exp(-self.zeta * self.omegan * self.t_range) * np.sin(
            self.omegad * self.t_range)  # displacement
        xd = np.hstack([[0], np.diff(xt) * fs])  # velocity
        sdof_reponse = np.hstack([[0], np.diff(xd) * fs])  # velocity
        return sdof_reponse
        # return 5000*xt


class Impulse():
    def __init__(self):
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


m = Measurement()
sp = SpeedProfile(speed_profile_type="sine")

bearing = Bearing(d, D, contactAngle, n_ball)
average_angular_distance_between_impulses = bearing.get_angular_distance_between_impulses("ball")
expected_number_of_impulses_during_measurement = sp.total_angle_traversed / average_angular_distance_between_impulses

imp = Impulse()
angular_distances_at_which_impulses_occur = imp.get_angular_distances_at_which_impulses_occur(
    expected_number_of_impulses_during_measurement,
    average_angular_distance_between_impulses,
    variance_factor_angular_distance_between_impulses)

# TODO: change the starting angle at which the fist fault occurs

def get_indexes_at_which_impulse_occur_for_single_measurement(angles_for_measurement,impulse_distances_in_measurement_interval):
    # Find the times corresponding to a given angle
    times_corresponding_to_angle = interp1d(angles_for_measurement, m.time)(impulse_distances_in_measurement_interval)

    # Find the indexes that is closest to the times at which the impulses occur
    indexes = find_nearest_index(m.time, times_corresponding_to_angle)
    return indexes


def get_impulses_distances_in_measurement_interval(impulse_distances_for_measurement,total_angle_traversed_for_measurement):
    # Discard the impulses that fall outside of the ranges of total angles traversed
    return impulse_distances_for_measurement[impulse_distances_for_measurement < total_angle_traversed_for_measurement]


indexes_at_which_impulses_occur = np.zeros((m.n_measurements,
                                            m.n_master_samples))  # Initialize empty array that show (master) samples where impulses will occur
for measurement in range(m.n_measurements):
    # For each separate measurement
    angles_for_measurement = sp.angles[measurement, :]
    impulse_distances_for_measurement = angular_distances_at_which_impulses_occur[measurement, :]
    total_angle_traversed_for_measurement = sp.total_angle_traversed[measurement]

    impulse_distances_in_measurement_interval = get_impulses_distances_in_measurement_interval(impulse_distances_for_measurement,total_angle_traversed_for_measurement)

    indexes = get_indexes_at_which_impulse_occur_for_single_measurement(angles_for_measurement,impulse_distances_in_measurement_interval)

    # Set the indexes where the impulses occur to 1 (Otherwise zero)
    indexes_at_which_impulses_occur[measurement, indexes] = 1

sys = SdofSys(m.master_frequency, k, zita, fn, 1)
transient = sys.get_transient_response()

convolved = scipy.signal.convolve2d(indexes_at_which_impulses_occur, transient.reshape(1, -1), mode="same")
# convolved2 = scipy.signal.convolve(indexes_at_which_impulses_occur, transient.reshape(1,-1), mode="same")
# These two options need to be looped over, can test performance
# plt.plot(np.convolve(transient,indexes_at_which_impulses_occur[0]))
# plt.plot(fftconvolve(transient,indexes_at_which_impulses_occur[0]))


# TODO: Potentially making error at the edges when convolving (need to include additional pulses than requested by the user to adress this issue)

measured = scipy.signal.decimate(convolved, 2, axis=1, ftype="fir")

# plt.figure()
# plt.plot(measured)
#
#
# plt.figure()
# plt.plot(indexes_at_which_impulses_occur[0]*np.max(convolved))
# # plt.plot(np.convolve(transient,indexes_at_which_impulses_occur[0]))
# # plt.plot(fftconvolve(transient,indexes_at_which_impulses_occur[0]))
# plt.plot(convolved[0])
# plt.plot(convolved2[0])

# time_at_which_impulses_occur[measurement,:] = interp1d(angle[measurement,:],m.time)(angular_distances_at_which_impulses_occur[measurement,:])

# print(interp_func(angular_distances_at_which_impulses_occur))
#
# print(interp_func)
# time_at_which_impulses_occur = interp_func(cumulative_impulse_distance[5:-4])
#
# print(time_at_which_impulses_occur)


# geometry_parameter = BearingGeometryParameter(d, D, contactAngle).get_geometry_parameter("outer")
# fr = SpeedProfile.get_rotation_frequency_as_function_of_time()
#
#
# Ltheta = len(fr)
#
# theta = (0:Ltheta-1)*2*pi/N; # Discretize a full revolution ie 0 - 2pi rads
#
# mean_delta_theta = 2*np.pi/(n*geometry_parameter) # Mean angular distance traveled between faults
#
# n_impulses = np.floor(theta[-1]/deltaThetaFault) # The total number of impulses expected during the measurement interval
#
# varDeltaTheta = (varianceFactor*mean_delta_theta)^2 # The variance in the angular distance between impulses
#
# # TODO: Why use variance factor?
# deltaThetaFault = np.sqrt(varDeltaTheta)*randn([1 n_impulses-1]) + meanDeltaTheta # This is the mean delta theta with normal noise
#
# thetaFault = [0 np.cumsum(deltaThetaFault)]; # Find the theta values at which an impulse is generated
#
# frThetaFault = interp1(theta,fr,thetaFault,'spline')  # Mapping between angle and rotation frequency, then evaluate at thetafault to get rotation frequency at the fault locations
#
# deltaTimp = deltaThetaFault ./ (2*pi*frThetaFault(2:end)); # Time distance between impulses?
#
# tTimp = [0 cumsum(deltaTimp)] # The time instances at which an impulse occurs
#
# L = floor(tTimp(end)*fs); % signal length # For some reason you now define a new signal length even though the user defined one in the beginning
#
# t = (0:L-1)/fs; # Get a linearly increasing time vector
#
# frTime = interp1(tTimp,frThetaFault,t,'spline'); # Create a mapping between the time at which the impulses occur and the frequency of rotation at a given impulse, then evaluate for all timesteps.
#
# deltaTimpIndex = round(deltaTimp*fs);
#
# errorDeltaTimp = deltaTimpIndex/fs - deltaTimp;
#
# indexImpulses = [1 cumsum(deltaTimpIndex)];
# # index = length(indexImpulses);
# # while indexImpulses(index)/fs > t(end)
# #     index = index - 1;
# # end
# # indexImpulses = indexImpulses(1:index);
# #
# # meanDeltaT = mean(deltaTimp);
# # varDeltaT = var(deltaTimp);
# # meanDeltaTimpOver = mean(deltaTimpIndex/fs);
# # varDeltaTimpOver = var(deltaTimpIndex/fs);
# #
# # x = zeros(1,L);
# # x(indexImpulses) = 1;
# #
# # % amplitude modulation
# # if strcmp(faultType,'inner')
# #     if length(fc) > 1,
# #         thetaTime = zeros(1,length(fr));
# #         for index = 2:length(fr),
# #             thetaTime(index) = thetaTime(index - 1) + (2*pi/N)/(2*pi*fr(index));
# #         end
# #         fcTime = interp1(thetaTime,fc,t,'spline');
# #         fdTime = interp1(thetaTime,fd,t,'spline');
# #         fmTime = interp1(thetaTime,fm,t,'spline');
# #         q = 1 + qAmpMod * cos(2*pi*fcTime.*t + 2*pi*fdTime.*(cumsum(cos(2*pi*fmTime.*t)/fs)));
# #     else
# #         q = 1 + qAmpMod * cos(2*pi*fc*t + 2*pi*fd*(cumsum(cos(2*pi*fm*t)/fs)));
# #     end
# #     x = q.*x;
# # end
# #
# # [sdofRespTime] = sdofResponse(fs,k,zita,fn,Lsdof);
# # x = fftfilt(sdofRespTime,x);
# #
# # L = length(x);
# # rng('default'); %set the random generator seed to default (for comparison only)
# # SNR = 10^(SNR_dB/10); %SNR to linear scale
# # Esym=sum(abs(x).^2)/(L); %Calculate actual symbol energy
# # N0 = Esym/SNR; %Find the noise spectral density
# # noiseSigma = sqrt(N0); %Standard deviation for AWGN Noise when x is real
# # nt = noiseSigma*randn(1,L);%computed noise
# # xNoise = x + nt; %received signal
