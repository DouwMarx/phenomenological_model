import matplotlib.pyplot as plt
import numpy as np


class AugmentedSES():
    """
    Augment an existing envelope spectrum towards the expected envelope spectrum for a given failure mode.
    """

    def __init__(self, healthy_ses=None, fs=38400, fault_frequency=74, percentage_of_freqs_to_decay_99_percent=0.5,
                 peak_magnitude=0.1):
        self.fs = fs

        self.signal_length = healthy_ses.shape[1] * 2 # Recover the initial signal length based on the length of the spectrum
        self.frequencies = np.fft.fftfreq(self.signal_length, 1 / fs)[0:int(self.signal_length / 2)]

        self.fault_frequency = fault_frequency

        self.healthy_ses = healthy_ses

        self.pp_sig = self.make_positive_periodic_signal()

        self.peak_magnitude = peak_magnitude

        self.percentage_of_freqs_to_decay_99_percent = percentage_of_freqs_to_decay_99_percent

    def make_positive_periodic_signal(self):
        """
        Create a positve periodic signal that can be modified to create the characteristic peaks in the squared envelope spectrum

        Returns
        -------

        """
        # return 0.5*(1 + np.sin(self.frequencies * 2 * np.pi / self.fault_frequency))
        return 0.5 * (1 + np.cos(self.frequencies * 2 * np.pi / self.fault_frequency))

    def sharp_peaks_at_fault_frequency(self):
        """
        Transform the positive periodic signal into sharp peaks at harmonics in the spectrum
        Returns
        -------

        """
        # return self.pp_sig ** 2
        # return self.pp_sig ** 4
        # return self.normalize(self.pp_sig ** 6)**6
        return self.normalize(self.pp_sig ** 8)
        # return self.normalize((-np.log(0.001+self.pp_sig)**4))
        # return np.log(self.pp_sig+0.1)# ** 6
        # return np.exp(self.pp_sig)

    def exponential_decay(self):
        """
        Let the peaks in the spectrum decay in magnitude with an increase in frequency

        Returns
        -------

        """
        freq_index_99_decay = int(self.percentage_of_freqs_to_decay_99_percent * self.signal_length * 0.5)
        # 0.01 means that the exponential has decayed 99% of its original magnitude
        scaling = np.log(0.01) / self.frequencies[freq_index_99_decay]
        decay = np.exp(scaling * self.frequencies)
        return decay

    @staticmethod
    def normalize(signal):
        """
        Normalize signal between 0 and 1

        Parameters
        ----------
        signal

        Returns
        -------

        """
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def get_augmented_ses(self):
        augmentation = self.peak_magnitude * self.exponential_decay() * self.normalize(
            self.sharp_peaks_at_fault_frequency())
        if self.healthy_ses is not None:
            augmentation = augmentation + self.healthy_ses

        return augmentation

    def show_augmented_ses(self):
        plt.figure()
        plt.plot(self.frequencies, self.get_augmented_ses())

# aug_obj = AugmentedSES(fault_frequency=60,percentage_of_freqs_to_decay_99_percent=0.1)
# aug_obj.show_augmented_ses()
