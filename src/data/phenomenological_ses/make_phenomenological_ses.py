import matplotlib.pyplot as plt
import numpy as np


class AugmentedSES():
    def __init__(self,healthy_ses = None,fs=38400, fault_frequency = 74,percentage_of_freqs_to_decay_99_percent = 0.1, peak_magnitude=1):
        self.fs = fs
        self.freqs = np.arange(0, int(fs / 2))
        self.fault_frequency = fault_frequency

        self.healthy_ses = healthy_ses

        self.pp_sig = self.make_positive_periodic_signal()

        self.peak_magnitude = peak_magnitude

        self.percentage_of_freqs_to_decay_99_percent = percentage_of_freqs_to_decay_99_percent

    def make_positive_periodic_signal(self):
        # return 0.5*(1 + np.sin(self.freqs * 2 * np.pi / self.fault_frequency))
        return 0.5*(1 + np.cos(self.freqs * 2 * np.pi / self.fault_frequency))

    def sharp_peaks_at_fault_frequency(self):
        # return self.pp_sig ** 2
        # return self.pp_sig ** 4
        return self.pp_sig ** 6
        # return np.log(self.pp_sig+0.1)# ** 6
        # return np.exp(self.pp_sig)

    def exponential_decay(self):

        # percentage = 0.01  # Percentage of the original amplitude to decay to # TODO: Add to kwargs
        # self.transient_duration = np.log(1 / percentage) / (self.zeta * self.omegan)

        freq_index_99_decay = int(self.percentage_of_freqs_to_decay_99_percent*self.fs*0.5)

        #np.exp(self.freqs[-1]*scaling) = 0.01
        scaling = np.log(0.01)/self.freqs[freq_index_99_decay]
        decay = np.exp(scaling * self.freqs)
        return decay


    @staticmethod
    def normalize(signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def get_augmented_ses(self):
        augmentation = self.peak_magnitude*self.exponential_decay()*self.normalize(self.sharp_peaks_at_fault_frequency())
        if self.healthy_ses is not None:
            augmentation = augmentation + self.healthy_ses

        return augmentation

    def show_augmented_ses(self):
        plt.figure()
        plt.plot(self.freqs, self.get_augmented_ses())


# aug_obj = AugmentedSES(fault_frequency=60,percentage_of_freqs_to_decay_99_percent=0.1)
# aug_obj.show_augmented_ses()


