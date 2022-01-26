import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from definitions import data_dir

# Loading the dataset
results_dictionary = np.load(data_dir.joinpath("generated_and_augmented.npy"),allow_pickle=True)[()]

def loop_through_mode_and_severity(results_dictionary,function_to_apply):
    for mode,dictionary in results_dictionary.items():
        for severity, measurements in dictionary.items():
            function_to_apply(mode,severity,results_dictionary)


def compare_augmented_and_generated_ses(mode,severity,results_dict):
    # Compare the augmented data with the generated data from the phenomenological model.
    plt.figure()
    plt.title("mode:" + mode +" | severity: " + severity)
    meas = results_dict[mode][severity]
    real_world = meas["envelope_spectrum"]
    augmented = meas["augmented_envelope_spectrum"]
    plt.plot(real_world["freq"],real_world["mag"][0],label ="Real world")
    plt.plot(augmented["freq"],augmented["mag"],label ="Augmented")

    plt.legend()

def show_example_time_signal(mode,severity,results_dict):
    # Compare the augmented data with the generated data from the phenomenological model.
    plt.figure()
    plt.title("mode:" + mode +" | severity: " + severity)
    meas = results_dict[mode][severity]
    sig = meas["time_domain"][0]
    plt.plot(sig)

loop_through_mode_and_severity(results_dictionary, compare_augmented_and_generated_ses)
# loop_through_mode_and_severity(results_dictionary, show_example_time_signal)

