"""
anomaly_datasets.py

This file defines several classes for generating and processing anomaly detection datasets,
particularly focusing on simulated signal datasets.

Classes:
    AnomalyDetectionDataset: Base class for anomaly detection datasets, providing data loading,
                             splitting, and standardization functionalities.
    SignalAnomalyDataset:  Abstract base class for signal-based anomaly detection datasets,
                           extending AnomalyDetectionDataset with signal segmentation capabilities.
    SimulatedSignalDataset: Abstract base class for simulated signal datasets, inheriting from
                            SignalAnomalyDataset and providing methods for generating signal components
                            (reference, faulty, noise).
    LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive: Concrete class inheriting from
                                                        SimulatedSignalDataset, generating a dataset
                                                        with low-frequency and high-frequency bandpass
                                                        impulsive signals representing healthy and faulty
                                                        conditions, respectively.
    LowFreqSineVSHighFreqSine: Concrete class inheriting from SimulatedSignalDataset, generating a dataset
                               with low-frequency and high-frequency sinusoidal signals representing
                               healthy and faulty conditions, respectively.

Example Usage:
    The `if __name__ == "__main__":` block at the end of this file provides a basic example
    of how to use these classes to generate datasets and access the split and standardized data.

Dependencies:
    - json
    - abc
    - numpy
    - scipy
    - torch (imported but not explicitly used in the provided code, may be relevant in a broader context)
    - scipy.optimize
    - scipy.signal
    - sklearn.model_selection
"""

from abc import ABC
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


class AnomalyDetectionDataset(object):
    """
    Base class for anomaly detection datasets.

    Provides functionalities for:
        - Data loading (abstract method, to be implemented by subclasses)
        - Splitting data into train, test, and validation sets
        - Standardizing data using various methods
        - Preprocessing data (abstract method, to be implemented by subclasses if needed)
        - Managing metadata about the dataset

    Attributes:
        split_seed (int): Random seed for data splitting, ensuring reproducibility.
        train_ratio (float): Ratio of data to be used for training.
        test_ratio (float): Ratio of data to be used for testing.
        validation_ratio (float): Ratio of data to be used for validation.
        standardization_method (str): Method for data standardization (e.g., "none", "standardize").
        use_subset_of_measurements (bool or int): If True, uses a subset of samples. If int, uses that many samples.
        name (str): Name of the dataset.
        data_dtype (str): Data type for the dataset (e.g., "float32").
        healthy_data (numpy.ndarray): Healthy data loaded from `load_data()`.
        faulty_data (dict): Dictionary of faulty data loaded from `load_data()`. Keys are fault mode names.
        feature_dimensions (int): Number of features (last dimension of data).
        meta_data (dict): Dictionary to store dataset metadata.

    Methods:
        __init__(self, ...): Initializes the AnomalyDetectionDataset.
        load_data(self): Abstract method to load data. Must be implemented by subclasses.
        get_split_and_standardized_sets(self): Splits data and applies standardization.
    """
    def __init__(self,
                 split_seed=1,
                 train_ratio=0.4,
                 test_ratio=0.3,
                 validation_ratio=0.3,
                 name=None,
                 data_dtype="float32"
                 ):

        """
        Initializes the AnomalyDetectionDataset.

        Args:
            split_seed (int, optional): Random seed for data splitting. Defaults to 1.
            train_ratio (float, optional): Ratio of data for training. Defaults to 0.4.
            test_ratio (float, optional): Ratio of data for testing. Defaults to 0.3.
            validation_ratio (float, optional): Ratio of data for validation. Defaults to 0.3.
            standardization_method (str, optional): Standardization method. Defaults to "none".
                                                    Options: "none", "remove_mean_over_time",
                                                             "remove_mean_over_time_and_rescale_by_reference_power",
                                                             "standardize_over_time", "standardize", "whiten_spectrum".
            name (str, optional): Name of the dataset. Defaults to None (nameless).
            data_dtype (str, optional): Data type of the dataset. Defaults to "float32".
        """

        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio

        self.data_dtype = data_dtype

        if name is not None:
            self.name = name
        else:
            self.name = "nameless"

        self.healthy_data, self.faulty_data = self.load_data() # Healthy data has shape (n_samples, n_channels, n_features) where n_channels is 1

        self.meta_data = {}
        self.meta_data.update({"name": self.name})
        self.meta_data.update({"train_ratio": self.train_ratio})
        self.meta_data.update({"test_ratio": self.test_ratio})
        self.meta_data.update({"validation_ratio": self.validation_ratio})

    def load_data(self):
        """
        Abstract method to load the dataset. Must be implemented by subclasses.

        Returns:
            tuple: A tuple containing:
                - healthy_data (numpy.ndarray): Healthy data.
                - faulty_data (dict): Dictionary of faulty data, where keys are fault mode names
                                       and values are numpy.ndarray of faulty data.
        """
        return None, None

    def get_split_and_standardized_sets(self):
        """
        Splits the loaded data into training, testing, and validation sets, and applies
        the specified standardization method.

        Returns:
            tuple: A tuple containing:
                - standardized_healthy_train (numpy.ndarray): Standardized healthy training data.
                - standardized_healthy_test (numpy.ndarray): Standardized healthy testing data.
                - standardized_healthy_val (numpy.ndarray or None): Standardized healthy validation data (None if validation_ratio is 0).
                - standardized_faulty_test (dict): Dictionary of standardized faulty testing data,
                                                     where keys are fault mode names.
        """
        # Check that ratios of each independent set add up to 1
        if self.train_ratio + self.validation_ratio + self.test_ratio != 1:
            raise ValueError("Dataset ratios should add up to 1")


        # Split between test and train (Where the training data can then be further split into train and validation)
        # Note that this random splitting requires that there is no overlap when segementing the signals, since this can lead to data contamination
        print("Caution, a random train-test-split is being used: No signal overlap supported")
        healthy_train, healthy_test = train_test_split(self.healthy_data, test_size=self.test_ratio,
                                                       random_state=self.split_seed)

        # If there is no validation set, there is no further splitting of the healthy training data
        if self.validation_ratio == 0:
            healthy_val = None
        else:
            # Split off the validation set from the training set (healthy_train is overwritten)
            healthy_train, healthy_val = train_test_split(healthy_train, test_size=self.validation_ratio / (
                        self.validation_ratio + self.train_ratio), random_state=self.split_seed)

        # Prepare the faulty test sets
        # (No faulty data used in training, multiple fault modes are possible i.e. multiple faulty test sets corresponding to single healthy dataset)
        faulty_test = {}
        for faulty_dataset_name, faulty_dataset in self.faulty_data.items():
            # Check that the faulty dataset has enough data to sample from
            #  TODO: Add some flexibility to allow an unbalanced test set if desired (For example, use average precision as a metric)
            faulty_test[faulty_dataset_name] = faulty_dataset

        return healthy_train, healthy_test, healthy_val, faulty_test

class SignalAnomalyDataset(AnomalyDetectionDataset):
    """
    Abstract base class for signal-based anomaly detection datasets.

    Extends AnomalyDetectionDataset with signal segmentation capabilities.
    Subclasses should implement `load_data()`

    Attributes:
        segment_length (int): Length of signal segments to be extracted.
        overlap_fraction (float): Fraction of overlap between consecutive segments (0.0 for no overlap).
        n_samples (int): Number of signal samples per class (healthy/faulty).
        signal_length (int): Total length of the generated signal (n_samples * segment_length).
        time (numpy.ndarray): Time vector for the signal.
        sampling_frequency (int): Sampling frequency of the signal in Hz.
        signal_duration (float): Total duration of the signal in seconds.
        meta_data (dict): Inherited and updated metadata dictionary.

    Methods:
        __init__(self, segment_length=12000, overlap_fraction=0.0, n_samples_per_class=1000,
                 sampling_frequency=1, **kwargs): Initializes SignalAnomalyDataset.
        cut_in_segments(self, signal, overlap_fraction=0.0): Cuts a signal into segments with specified overlap.
        add_gaussian_noise_at_snr(self, x, signal_to_noise_ratio=1): Adds Gaussian noise to a signal at a given SNR.
    """
    def __init__(self,
                 segment_length=12000,
                 overlap_fraction=0.0,
                 n_samples_per_class=1000,
                 sampling_frequency=1,
                 **kwargs,
                 ):
        """
        Initializes the SignalAnomalyDataset.

        Args:
            segment_length (int, optional): Length of each signal segment. Defaults to 12000.
            overlap_fraction (float, optional): Overlap fraction between segments. Defaults to 0.0 (no overlap).
            n_samples_per_class (int, optional): Number of samples per class. Defaults to 1000.
            sampling_frequency (int, optional): Sampling frequency in Hz. Defaults to 1.
            **kwargs: Keyword arguments passed to the parent class AnomalyDetectionDataset.
        """
        self.segment_length = segment_length
        self.overlap_fraction = overlap_fraction
        self.n_samples = n_samples_per_class
        self.signal_length = n_samples_per_class * self.segment_length

        self.time = np.arange(self.signal_length) / sampling_frequency  # [s]
        self.sampling_frequency = sampling_frequency  # [Hz]
        self.signal_duration = self.time[-1]  # [s]

        super().__init__(**kwargs)

        self.meta_data.update({
            # "signal_duration": self.signal_duration, # Redundant info as signal_length and sampling_frequency are given
            "sampling_frequency": sampling_frequency,
            "cut_signal_length": segment_length,
        }
        )

    def cut_in_segments(self, signal, overlap_fraction=0.0):
        """
        Cuts a given signal into segments of `self.segment_length` with a specified overlap.

        Args:
            signal (numpy.ndarray): 1D signal to be segmented.
            overlap_fraction (float, optional): Fraction of overlap between segments. Defaults to 0.0.

        Returns:
            numpy.ndarray: Array of signal segments. Shape: (n_segments, segment_length).
        """
        data = []
        step = int(self.segment_length * (1 - overlap_fraction))
        for i in range(0, len(signal), step):
            if i + self.segment_length <= len(signal):
                data.append(signal[i:i + self.segment_length])
        return np.array(data)


class SimulatedSignalDataset(SignalAnomalyDataset, ABC):
    """
    Abstract base class for simulated signal datasets.

    Inherits from SignalAnomalyDataset and provides abstract methods for generating
    reference, faulty, and noise signal components. Concrete subclasses must implement
    `get_reference_component()` and `get_faulty_component()` to define the signal generation process.

    Attributes:
        noise_std (float): Standard deviation of the generated noise.
        meta_data (dict): Inherited and updated metadata dictionary.

    Methods:
        __init__(self, noise_std=1, **kwargs): Initializes SimulatedSignalDataset.
        get_reference_component(self, **kwargs): Abstract method to generate the reference signal component.
                                                Must be implemented by subclasses.
        get_faulty_component(self): Abstract method to generate the faulty signal component.
                                     Must be implemented by subclasses.
        get_noise_component(self, siglen=None): Generates a noise signal component.
        load_data(self): Loads and processes data by generating reference, faulty, and noise components,
                       cutting into segments, and applying preprocessing.
        show_components(self, domain="time"): Visualizes the individual signal components and their combinations
                                             in either the time or frequency domain (requires plotly).
        show_faulty_component_envelope_spectrum(self): Calculates and displays the envelope spectrum
                                                      of the faulty component (requires plotly).
    """
    def __init__(self, noise_std=1, **kwargs):
        """
        Initializes the SimulatedSignalDataset.

        Args:
            noise_std (float, optional): Standard deviation of the noise component. Defaults to 1.
            **kwargs: Keyword arguments passed to the parent class SignalAnomalyDataset.
        """
        self.noise_std = noise_std
        super().__init__(**kwargs)

    def get_reference_component(self,**kwargs):
        """
        Abstract method to generate the reference (healthy) signal component.
        Must be implemented by concrete subclasses.

        Returns:
            numpy.ndarray: Reference signal component.
        """
        pass

    def get_faulty_component(self):
        """
        Abstract method to generate the faulty signal component.
        Must be implemented by concrete subclasses.

        Returns:
            numpy.ndarray: Faulty signal component.
        """
        pass

    def get_noise_component(self, siglen=None):
        """
        Generates a Gaussian white noise signal component.

        Args:
            siglen (int, optional): Length of the noise signal to generate. If None, uses `self.signal_length`.
                                     Defaults to None.

        Returns:
            numpy.ndarray: Noise signal component.
        """
        if siglen == None:
            return np.random.randn(self.signal_length) * self.noise_std
        elif type(siglen) == int:
            return np.random.randn(siglen) * self.noise_std
        else:
            raise ValueError("siglen should be an integer")

    def load_data(self):
        """
        Loads and processes data by generating reference, faulty, and noise components,
        cutting the signals into segments, and applying preprocessing.

        Returns:
            tuple: A tuple containing:
                - healthy_data (numpy.ndarray): Processed healthy data.
                - faulty_data (dict): Dictionary of processed faulty data, with fault mode name as key.
        """
        # Idea is that the measurement noise level is always with std 1, and the signal amplitudes are scaled
        # Use a different random seed for each case
        np.random.seed(0)
        x0_noise = self.get_noise_component()
        x0 = self.get_reference_component() + x0_noise
        print("reference kurtosis ", np.mean(x0 ** 4) / np.mean(x0 ** 2) ** 2)

        np.random.seed(1)
        x_noise = self.get_noise_component()
        x =  self.get_reference_component()
        x += self.get_faulty_component()
        x += x_noise
        print("faulty kurtosis ", np.mean(x ** 4) / np.mean(x ** 2) ** 2)

        x_healthy = self.cut_in_segments(x0,overlap_fraction=self.overlap_fraction)
        x_faulty = self.cut_in_segments(x,overlap_fraction=self.overlap_fraction)

        # Add a channel dimension
        x_healthy = x_healthy.reshape(x_healthy.shape[0], 1, x_healthy.shape[1])
        x_faulty = x_faulty.reshape(x_faulty.shape[0], 1, x_faulty.shape[1])

        return x_healthy, {"Impulsive Bandpass Noise": x_faulty}

    def show_components(self,domain="time"):
        """
        Shows a subplot of each signal component and their combinations.
        Requires plotly for visualization.

        Args:
            domain (str, optional): Domain for visualization ("time" or "frequency"). Defaults to "time".

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Get the different components
        np.random.seed(0)
        reference = self.get_reference_component()[:self.segment_length]
        noise_reference = self.get_noise_component()[:self.segment_length]

        np.random.seed(1)
        faulty = self.get_faulty_component()[:self.segment_length]
        noise_faulty = self.get_noise_component()[:self.segment_length]

        reference_plus_noise = reference + noise_reference
        faulty_plus_reference_plus_noise = reference + faulty + noise_faulty

        if domain == "time":
            independent_variable = np.arange(self.segment_length) / self.sampling_frequency
        elif domain == "frequency":
            independent_variable = np.fft.rfftfreq(self.segment_length, 1 / self.sampling_frequency)
            reference = np.abs(np.fft.rfft(reference))
            faulty = np.abs(np.fft.rfft(faulty))
            noise_reference = np.abs(np.fft.rfft(noise_reference))
            reference_plus_noise = np.abs(np.fft.rfft(reference_plus_noise))
            faulty_plus_reference_plus_noise = np.abs(np.fft.rfft(faulty_plus_reference_plus_noise))
        else:
            raise ValueError("Unknown domain {}".format(domain))

        fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                            vertical_spacing=0.0)  # Space between subplots maximal when vertical_spacing = 0.0
        fig.add_trace(go.Scatter(x=independent_variable, y=reference, line=dict(color="black")), row=1, col=1)
        fig.add_trace(go.Scatter(x=independent_variable, y=faulty, line=dict(color="black")), row=2, col=1)
        fig.add_trace(go.Scatter(x=independent_variable, y=noise_reference, line=dict(color="black")), row=3, col=1)
        fig.add_trace(go.Scatter(x=independent_variable, y=reference_plus_noise, line=dict(color="black")), row=4, col=1)
        fig.add_trace(go.Scatter(x=independent_variable, y=faulty_plus_reference_plus_noise, line=dict(color="black")), row=5, col=1)

        # Make all lines thin and dont show the labels
        for trace in fig.data:
            trace.line.width = 0.5
            trace.showlegend = False

        annotation_dict = dict(
            text="1",  # Your large number
            xref="x domain",
            yref="y domain",
            x=0.03,
            y=1,
            showarrow=False,
            font=dict(size=20),
            # Solid white background
            bgcolor="white",
            # bordercolor="black",
            # borderwidth=3,
        )

        annotation_dict.update({"text": 'Reference Component (1)'})
        # fig.update_yaxes(title_text=r'$\text{Reference Component }  [m/s^2]$', row=1, col=1)

        fig.add_annotation(**annotation_dict, row=1, col=1)

        annotation_dict.update({"text": 'Faulty Component: (2)'})
        # fig.update_yaxes(title_text=r'$\text{Faulty Component }  [m/s^2]$', row=2, col=1)
        fig.add_annotation(**annotation_dict, row=2, col=1)

        annotation_dict.update({"text": 'Noise Component: (3)'})
        # fig.update_yaxes(title_text=r'$\text{Noise Component }  [m/s^2]$', row=3, col=1)
        fig.add_annotation(**annotation_dict, row=3, col=1)

        # annotation_dict.update({"text": r'$\text{Reference Condition } \quad (1)+(3) $'})
        annotation_dict.update({"text": 'Reference Condition: (1) + (3)'})
        # fig.update_yaxes(title_text= r'$\text{Reference Condition } (\mathcal{H}_0) \text{ } [m/s^2]$', row=4, col=1)
        fig.add_annotation(**annotation_dict, row=4, col=1)

        annotation_dict.update({"text": 'Faulty Condition: (1) + (2) + (3)'})
        # fig.update_yaxes(title_text= r'$\text{Faulty Condition } (\mathcal{H}_1)) \text{ }  [m/s^2]$', row=5, col=1)
        # Make the y-axis label horizontal
        # fig.update_yaxes(title_standoff=0, row=5, col=1)

        fig.add_annotation(**annotation_dict, row=5, col=1)

        # Add a master axis to the left of the figures
        # fig.update_yaxes(title_text=r'$\text{Signal Component }  [m/s^2]$', row=3, col=1)
        fig.update_yaxes(title_text=r'$\Large{\text{Signal Component }  [\text{m/s}^{2}]}$', row=3, col=1)

        ticksize = 16
        # Make the tick labels smaller
        fig.update_layout(
            yaxis=dict(tickfont=dict(size=ticksize)),
            yaxis2=dict(tickfont=dict(size=ticksize)),
            yaxis3=dict(tickfont=dict(size=ticksize)),
            yaxis4=dict(tickfont=dict(size=ticksize)),
            yaxis5=dict(tickfont=dict(size=ticksize)),
        )

        fig.update_layout(
            xaxis=dict(tickfont=dict(size=ticksize)),
            xaxis2=dict(tickfont=dict(size=ticksize)),
            xaxis3=dict(tickfont=dict(size=ticksize)),
            xaxis4=dict(tickfont=dict(size=ticksize)),
            xaxis5=dict(tickfont=dict(size=ticksize)),
        )

        if domain == "time":
            x_axis_title = r'$\Large{\text{Time [s]}}$'
        elif domain == "frequency":
            x_axis_title = r'$\Large{\text{Frequency [Hz]}}$'

        fig.update_layout(
            title="Signal Components",
            # xaxis5_title=r'$\text{Time [s]}$',
            xaxis5_title=x_axis_title
        )

        fig.show()
        return fig

    def show_faulty_component_envelope_spectrum(self):
        """
        Calculates and displays the envelope spectrum of the faulty component.
        Requires plotly for visualization.

        The envelope spectrum is obtained by squaring the faulty component signal and
        then computing its Fourier transform.

        Optionally, if `self.ground_truth_fault_frequency` is defined, it will be shown
        as a vertical dashed red line on the plot.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        signal = self.get_faulty_component()
        x_squared = signal ** 2
        x_squared_fft = np.fft.rfft(x_squared)
        freq = np.fft.rfftfreq(len(x_squared), 1 / self.sampling_frequency)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=np.abs(x_squared_fft), mode='lines', name='Amplitude Spectrum'))
        fig.update_layout(
            xaxis_title="Frequency [Hz]",
            yaxis_title="Amplitude",
            title="Envelope Spectrum of Faulty Component"
        )

        # Show the datasets ground truth fault frequency as a vertical line
        try:
            fig.add_shape(
                dict(
                    type="line",
                    x0=self.ground_truth_fault_frequency,
                    y0=0,
                    x1=self.ground_truth_fault_frequency,
                    y1=max(np.abs(x_squared_fft)),
                    line=dict(color="red", dash="dash")
                )
            )
        except AttributeError:
            print("Ground truth fault frequency not available for dataset")

        fig.show()
        return fig



class LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive(SimulatedSignalDataset):
    """
    Dataset class generating low-frequency bandpass impulsive signals for healthy conditions
    and high-frequency bandpass impulsive signals for faulty conditions.

    This dataset simulates a scenario where a fault introduces impulsive noise in a different
    frequency band than the normal operating noise.

    Inherits from SimulatedSignalDataset.

    Attributes:
        f_ref (float): Frequency of impulses in the reference (healthy) component (Hz).
        f_fault (float): Frequency of impulses in the faulty component (Hz).
        peak_fault (float): Peak amplitude of faulty impulses.
        peak_ref (float): Peak amplitude of reference impulses.
        reference_epsilon (float): Epsilon parameter for reference impulse generation.
        faulty_epsilon (float): Epsilon parameter for faulty impulse generation.
        reference_band (list): Frequency band [lowcut, highcut] (fraction of Nyquist) for reference component.
        faulty_band (list): Frequency band [lowcut, highcut] (fraction of Nyquist) for faulty component.
        noise_std (float): Standard deviation of additive Gaussian noise.
        meta_data (dict): Inherited and updated metadata dictionary, including dataset-specific parameters.

    Methods:
        __init__(self, ...): Initializes LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive dataset.
        get_bandpass_noise(self, lowcut, highcut, order=7): Generates bandpass filtered Gaussian noise.
        get_impulses(self, amplitude, frequency, epsilon=1e-6): Generates impulsive signal using a cosine-based function.
        get_reference_component(self): Generates the reference signal component (low-frequency bandpass impulsive).
        get_faulty_component(self): Generates the faulty signal component (high-frequency bandpass impulsive).
    """
    def __init__(self,
                 peak_fault=2,
                 peak_ref=5,
                 reference_band=[0, 0.7],  # Band as fraction of nyquist
                 faulty_band=[0.6, 0.8],  # Band as fraction of nyquist (7000Hz is around 7000/(20480/2) = 0.68)
                 f_fault=108.1,  # Hz (Bearing fault frequency, BPFO)
                 f_ref=273.7,  # Hz (Gear mesh frequency) @ 1500 RPM
                 # To ensure 10 fault events per segments we require a segment length
                 # 108 times per second is 108/20480  times per sample
                 # We need 10 events per segment, so we need 10*20480/108 samples per segment
                 # That is roughly 2000 samples per segment
                 n_samples_per_class=100,  #
                 segment_length=800, #  1024, # 5120,
                 #  int(10240), #  int(20480), # int(10240), #  int(5120),#  int(4096), #    int(10240),  # Samples
                 sampling_frequency=20480,
                 reference_epsilon=1,
                 faulty_epsilon=1e-2,
                 noise_std=1,
                 **kwargs):
        """
        Initializes the LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive dataset.

        Args:
            peak_fault (float, optional): Peak amplitude of faulty impulses. Defaults to 2.
            peak_ref (float, optional): Peak amplitude of reference impulses. Defaults to 5.
            reference_band (list, optional): Frequency band for reference component [lowcut, highcut] (fraction of Nyquist). Defaults to [0, 0.7].
            faulty_band (list, optional): Frequency band for faulty component [lowcut, highcut] (fraction of Nyquist). Defaults to [0.6, 0.8].
            f_fault (float, optional): Frequency of faulty impulses (Hz). Defaults to 108.1.
            f_ref (float, optional): Frequency of reference impulses (Hz). Defaults to 273.7.
            n_samples_per_class (int, optional): Number of samples per class. Defaults to 100.
            segment_length (int, optional): Length of signal segments. Defaults to 800.
            sampling_frequency (int, optional): Sampling frequency in Hz. Defaults to 20480.
            reference_epsilon (float, optional): Epsilon parameter for reference impulse generation. Defaults to 1.
            faulty_epsilon (float, optional): Epsilon parameter for faulty impulse generation. Defaults to 1e-2.
            noise_std (float, optional): Standard deviation of additive noise. Defaults to 1.
            **kwargs: Keyword arguments passed to the parent class SimulatedSignalDataset.
        """

        # Parameters are explicity set so that the same dataset can be generated multiple times without changing the parameters
        self.f_ref = f_ref
        self.f_fault = f_fault
        self.peak_fault = peak_fault
        self.peak_ref = peak_ref
        self.reference_epsilon = reference_epsilon
        self.faulty_epsilon = faulty_epsilon
        self.reference_band = reference_band
        self.faulty_band = faulty_band
        self.noise_std = noise_std

        super().__init__(
            name="impulsivesim",
            noise_std=noise_std,
            n_samples_per_class=n_samples_per_class,
            segment_length=segment_length,
            sampling_frequency=sampling_frequency,
            **kwargs)

        self.meta_data.update({
            "dataset_name": "Low frequency bandpass impulsive vs high frequency bandpass impulsive",
            "f_ref": f_ref,
            "f_fault": f_fault,
            "a_fault": peak_fault,
            "a_ref": peak_ref,
            "reference_band": reference_band,
            "faulty_band": faulty_band,
            "faulty_band_hz": list(np.array(faulty_band)*sampling_frequency / 2),
            "reference_band_hz": list(np.array(reference_band)*sampling_frequency / 2),
        })

    def get_bandpass_noise(self, lowcut, highcut, order=7):
        """
        Generates bandpass filtered Gaussian white noise.

        Args:
            lowcut (float): Low-cut frequency as a fraction of the Nyquist frequency (0 to 1).
            highcut (float): High-cut frequency as a fraction of the Nyquist frequency (0 to 1).
            order (int, optional): Order of the Butterworth filter. Defaults to 7.

        Returns:
            numpy.ndarray: Bandpass filtered noise signal.
        """
        if lowcut == 0 and highcut == 1:
            x = np.random.randn(self.signal_length)
            return x / np.std(x)  # Standardize the signal to have unit variance

        if 1 > lowcut > 0 and 1 > highcut > 0:
            b, a = butter(order, [lowcut, highcut], btype='bandpass')
        elif lowcut == 0:
            b, a = butter(order, highcut, btype='lowpass')
        elif highcut == 1:
            b, a = butter(order, lowcut, btype='highpass')
        else:
            raise ValueError("Unknown bandpass filter type")
        x = filtfilt(b, a, np.random.randn(self.signal_length))
        x = x / np.std(x)  # Standardize the signal to have unit variance
        return x

    def get_impulses(self, amplitude, frequency, epsilon=1e-6):
        """
        Generates an impulsive signal using a cosine-based function.

        This function creates impulses with a specific frequency and amplitude using
        a modulated cosine wave. The `epsilon` parameter controls the sharpness of the impulses.

        Args:
            amplitude (float): Amplitude of the impulses.
            frequency (float): Frequency of the impulses (Hz).
            epsilon (float, optional): Parameter controlling impulse sharpness. Smaller values lead to sharper impulses.
                                       Defaults to 1e-6.

        Returns:
            numpy.ndarray: Impulsive signal.
        """

        # Elegant version, but does not completely reach zero
        # return amplitude*epsilon/(1+ epsilon - np.cos(2*np.pi*frequency*self.time)) # For the cos version you need to remember the negative sign, but the first impulse is at zero

        # Less elegant version, but reaches zero
        """
        Latex

        s(t) = \frac{1}{2}\cos(2\pi f t) + \frac{1}{2}
        m(t) = \frac{A \epsilon s(t)}{1 + \epsilon - s(t)}
        """
        random_phase = np.random.rand() * 2 * np.pi
        # print("Print random phase", random_phase)

        s = 0.5 * np.cos(2 * np.pi * frequency * self.time + random_phase) + 0.5
        return amplitude * epsilon * s / (1 + epsilon - s)

    def get_reference_component(self):
        """
        Generates the reference (healthy) signal component.

        This component consists of low-frequency bandpass noise modulated by impulses.

        Returns:
            numpy.ndarray: Reference signal component.
        """
        sig =  self.get_impulses(self.peak_ref, self.f_ref, epsilon=self.reference_epsilon) * self.get_bandpass_noise(
            self.reference_band[0], self.reference_band[1])
        sig = sig/np.std(sig) # Standardize the signal to have unit variance
        return sig

    def get_faulty_component(self):
        """
        Generates the faulty signal component.

        This component consists of high-frequency bandpass noise modulated by impulses.

        Returns:
            numpy.ndarray: Faulty signal component.
        """
        sig = self.get_impulses(self.peak_fault, self.f_fault, epsilon=self.faulty_epsilon) * self.get_bandpass_noise(
            self.faulty_band[0], self.faulty_band[1])
        sig = sig/np.std(sig) # Standardize the signal to have unit variance
        return sig


class LowFreqSineVSHighFreqSine(SimulatedSignalDataset):
    """
    Dataset class generating low-frequency sinusoidal signals for healthy conditions
    and high-frequency sinusoidal signals for faulty conditions.

    This dataset simulates a scenario where a fault introduces a sinusoidal signal at a different
    frequency than the normal operating signal.

    Inherits from SimulatedSignalDataset.

    Attributes:
        f_ref (float): Frequency of the reference (healthy) sine wave (fraction of Nyquist).
        f_fault (float): Frequency of the faulty sine wave (fraction of Nyquist).
        peak_fault (float): Amplitude of the faulty sine wave.
        include_interference_at_fault_freq (bool): If True, includes a sine wave at the fault frequency in the reference signal.
        noise_std (float): Standard deviation of additive Gaussian noise.
        meta_data (dict): Inherited and updated metadata dictionary, including dataset-specific parameters.

    Methods:
        __init__(self, ...): Initializes LowFreqSineVSHighFreqSine dataset.
        get_sine(self, frequency): Generates a sine wave at the given frequency.
        get_reference_component(self): Generates the reference signal component (low-frequency sine wave).
        get_faulty_component(self): Generates the faulty signal component (high-frequency sine wave).
    """
    def __init__(self,
                 f_ref=3 / 20,  # Frequency as fraction of Nyquist
                 f_fault=8 / 20,
                 n_samples_per_class=100,
                 segment_length=20,
                 sampling_frequency=1,
                 peak_fault=0.1,
                 peak_ref=1,
                 include_interference_at_fault_freq=False,
                 noise_std=1,
                 **kwargs):
        """
        Initializes the LowFreqSineVSHighFreqSine dataset.

        Args:
            f_ref (float, optional): Frequency of reference sine wave (fraction of Nyquist). Defaults to 3/20.
            f_fault (float, optional): Frequency of faulty sine wave (fraction of Nyquist). Defaults to 8/20.
            n_samples_per_class (int, optional): Number of samples per class. Defaults to 100.
            segment_length (int, optional): Length of signal segments. Defaults to 20.
            sampling_frequency (int, optional): Sampling frequency in Hz. Defaults to 1.
            peak_fault (float, optional): Amplitude of the faulty sine wave. Defaults to 0.1.
            include_interference_at_fault_freq (bool, optional): Include interference at fault frequency in reference signal. Defaults to False.
            noise_std (float, optional): Standard deviation of additive noise. Defaults to 1.
            **kwargs: Keyword arguments passed to the parent class SimulatedSignalDataset.
        """

        self.f_ref = f_ref
        self.f_fault = f_fault
        self.include_interference_at_fault_freq = include_interference_at_fault_freq

        self.peak_fault = peak_fault
        self.peak_ref = peak_ref

        super().__init__(
            name="sinesim",
            noise_std=noise_std,
            n_samples_per_class=n_samples_per_class,
            segment_length=segment_length,
            sampling_frequency=sampling_frequency,
            **kwargs)

        self.meta_data.update({
            "dataset_name": "Low frequency sine vs high frequency sine",
            "f_ref": f_ref,
            "f_fault": f_fault,
        })

    def get_sine(self, frequency):
        """
        Generates a sine wave at the specified frequency.

        Args:
            frequency (float): Frequency of the sine wave (fraction of Nyquist).

        Returns:
            numpy.ndarray: Sine wave signal.
        """
        return np.sin(2 * np.pi * frequency * self.time)

    def get_reference_component(self):
        """
        Generates the reference (healthy) signal component.

        This component consists of a low-frequency sine wave.
        If `include_interference_at_fault_freq` is True, it also includes a sine wave at the fault frequency.

        Returns:
            numpy.ndarray: Reference signal component.
        """
        if self.include_interference_at_fault_freq:
            return self.get_sine(self.f_ref)*self.peak_ref + self.get_sine(self.f_fault)*self.peak_ref
        else:
            return self.get_sine(self.f_ref)*self.peak_ref

    def get_faulty_component(self):
        """
        Generates the faulty signal component.

        This component consists of a high-frequency sine wave, scaled by `self.faulty_amplitude`.

        Returns:
            numpy.ndarray: Faulty signal component.
        """
        return self.get_sine(self.f_fault) * self.peak_fault


if __name__ == "__main__":
    # Example usage of the LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive dataset
    impulsive_dataset = LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive(
        peak_fault=2,
        peak_ref=5,
        reference_band=[0, 0.7],  # Band as fraction of Nyquist
        faulty_band=[0.6, 0.8],  # Band as fraction of Nyquist
        n_samples_per_class=50,  # Reduced samples for quick example
        segment_length=1000,      # Reduced segment length
        sampling_frequency=10240, # Reduced sampling frequency
        train_ratio=0.6,
        test_ratio=0.2,
        validation_ratio=0.2,
    )
    healthy_train, healthy_test, healthy_val, faulty_test = impulsive_dataset.get_split_and_standardized_sets()
    impulsive_dataset.show_components()

    print("Impulsive Dataset:")
    print("Healthy Train data shape:", healthy_train.shape)
    print("Healthy Test data shape:", healthy_test.shape)
    if healthy_val is not None:
        print("Healthy Validation data shape:", healthy_val.shape)
    print("Faulty Test data keys:", faulty_test.keys())
    for fault_mode, data in faulty_test.items():
        print(f"Faulty Test data shape ({fault_mode}):", data.shape)


    # Example usage of the LowFreqSineVSHighFreqSine dataset
    sine_dataset = LowFreqSineVSHighFreqSine(
        peak_fault=0.1,
        peak_ref=1,
        f_ref=3 / 10,  # Frequency as fraction of Nyquist
        f_fault=8 / 10,
        n_samples_per_class=50, # Reduced samples for quick example
        segment_length=1000,     # Reduced segment length
        sampling_frequency=64,  # Reduced sampling frequency
        train_ratio=0.5,
        test_ratio=0.5,
        validation_ratio=0.0, # No validation for simplicity in this example
    )
    healthy_train_sine, healthy_test_sine, healthy_val_sine, faulty_test_sine = sine_dataset.get_split_and_standardized_sets()
    sine_dataset.show_components()

    print("\nSine Dataset:")
    print("Healthy Train data shape:", healthy_train_sine.shape)
    print("Healthy Test data shape:", healthy_test_sine.shape)
    if healthy_val_sine is not None:
        print("Healthy Validation data shape:", healthy_val_sine.shape) # Will be None in this example
    print("Faulty Test data keys:", faulty_test_sine.keys())
    for fault_mode, data in faulty_test_sine.items():
        print(f"Faulty Test data shape ({fault_mode}):", data.shape)
