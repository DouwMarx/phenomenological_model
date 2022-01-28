from src.data.phenomenological_bearing_model.make_data import PyBearingDataset
import numpy as np
from src.data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from definitions import data_dir
from sklearn.decomposition import PCA
from definitions import data_dir
from src.utils.sigproc import env_spec


def compute_features_from_time_domain_signal(signal, fs):
    """
    Computes features from the time domain signal. In this case the envelope spectrum is computed

    Parameters
    ----------
    mode
    severity
    measurement

    Returns
    -------

    """
    # env = envelope(signal) # The envelope of the signal can also be added to the computed features
    freq, mag, phase = env_spec(signal, fs=fs)

    computed_features = {"envelope_spectrum": {"freq": freq,
                                               "mag": mag,
                                               "phase": phase
                                               }}

    return computed_features


def compute_signal_augmentation(mode, severity, results_dict):
    """
    Augments healthy data towards a faulty state for a given failure mode.

    Parameters
    ----------
    mode
    severity
    results_dict

    Returns
    -------

    """

    # Retrieve some meta data required for creating the augmented signal
    meta_data = results_dict[mode][severity]["meta_data"]
    fs = meta_data["sampling_frequency"]
    expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

    # Using all of the healthy ses as input means that the augmented dataset will have the noise of the training set
    # However, among the augmented dataset the "signal" will be identical
    # notice the "0" meaning that we are using healthy data
    healthy_ses = results_dict[mode]["0"]["envelope_spectrum"]["mag"]  # [0] # Use the first one

    ases = AugmentedSES(healthy_ses=healthy_ses, fs=fs, fault_frequency=expected_fault_frequency,
                        peak_magnitude=0.03)  # TODO: Fix peak magnitude, providing augmentation parameters?
    envelope_spectrum = ases.get_augmented_ses()

    return {"augmented_envelope_spectrum": {"freq": ases.freqs,
                                            "mag": envelope_spectrum}}


def add_processed_and_augmented_data(data_dict):
    """
    Adds features derived from the time domain signal as well as augmentation of the healthy data.
    This function is ran before models are trained and the encodings are added to dictionary.
    Parameters
    ----------
    data_dict

    Returns
    -------

    """
    for mode, dictionary in data_dict.items():
        for severity, measurements in dictionary.items():
            # Add the processed signal i.e. SES
            signal = measurements["time_domain"]
            fs = data_dict[mode][severity]["meta_data"]["sampling_frequency"]

            processed = compute_features_from_time_domain_signal(signal, fs)
            data_dict[mode][severity].update(processed)

            # Add the augmented signals
            augmented = compute_signal_augmentation(mode, severity, data_dict)
            data_dict[mode][severity].update(augmented)

    return data_dict


def limit_frequency_components(arr, fraction_of_spectrum_to_use=0.25):
    """
    Removes the DC component of the Squared envelope spectrum.
    Furthermore, it uses only a fraction of the total spectrum
    Parameters
    ----------
    fraction_of_spectrum_to_use
    arr

    Returns
    -------

    """
    siglen = arr.shape[1]
    use = int(siglen * fraction_of_spectrum_to_use)
    return arr[:, 1:use]


def compute_encodings(data):
    """
    Train unsupervised models and compute the encodings that they result in.

    Parameters
    ----------
    data

    Returns
    -------

    """
    # Define models
    model_healthy_only = PCA(2)
    model_healthy_only.name = "healthy_only"
    model_healthy_and_augmented = PCA(2)
    model_healthy_and_augmented.name = "healthy_and_augmented"

    # Set up training data
    # Healthy data only
    all_healthy = [data[mode]["0"]["envelope_spectrum"]["mag"] for mode in list(data.keys())]
    healthy_train = limit_frequency_components(np.vstack(
        all_healthy))  # Healthy data from different "modes" even though modes don't technically exist when healthy

    # Healthy dan augmented data
    all_augmented_modes = [data[mode]["1"]["augmented_envelope_spectrum"]["mag"] for mode in list(data.keys())]
    augmented_and_healthy_train = limit_frequency_components(np.vstack(all_healthy + all_augmented_modes))

    # Train the models
    model_healthy_only.fit(healthy_train)
    model_healthy_and_augmented.fit(augmented_and_healthy_train)

    # List of trained models
    models = [model_healthy_only, model_healthy_and_augmented]

    # Loop through all failure modes and severities.
    # For both the augmented and actual data, compute the expected encoding for each of the trained models.
    for mode_name, mode_data in data.items():
        for severity_name, severity_data in mode_data.items():
            data_type_dict = {}  # Data type refers to either real or augmented
            for data_type in ["envelope_spectrum", "augmented_envelope_spectrum"]:
                model_type_dict = {}
                for model in models:
                    encoding = model.transform(
                        limit_frequency_components(severity_data[data_type]["mag"]))

                    # Update the dictionary with the encodings
                    model_type_dict.update({model.name: encoding})
                data_type_dict.update({data_type + "_encoding": model_type_dict})
            data[mode_name][severity_name].update(data_type_dict)
    return data


def run_data_and_encoding_pipeline(data_name,quik_iter = True):

    # Create phenomenological model object
    o = PyBearingDataset(n_severities=10, failure_modes=["ball", "inner", "outer"], quick_iter=quik_iter) # TODO: Drive these parameters with governing yaml file

    # Generate phenomenological data
    results_dictionary = o.make_measurements_for_different_failure_mode()

    # Process the time series data and add augmented data
    results_dictionary = add_processed_and_augmented_data(results_dictionary)

    # Train unsupervised models and add encodings
    results_dictionary = compute_encodings(results_dictionary)

    if quik_iter:
        data_name = data_name + "quick_iter"
    np.save(data_dir.joinpath(data_name + ".npy"), results_dictionary, allow_pickle=True)


if __name__ == "__main__":
    run_data_and_encoding_pipeline("generated_and_augmented")