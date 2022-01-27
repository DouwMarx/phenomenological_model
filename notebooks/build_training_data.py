from src.data.phenomenological_bearing_model.make_data import PyBearingDatasest,envelope,env_spec
import numpy as np
from src.data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from definitions import data_dir
from sklearn.decomposition import PCA
from definitions import data_dir


def processed_signals(mode,severity,measurement):
    signal = measurement["time_domain"]
    fs=results_dictionary[mode][severity]["meta_data"]["sampling_frequency"]
    # env = envelope(signal)
    # print(np.isnan(env).sum())
    freq, mag, phase = env_spec(signal, fs=fs)
    # print(np.isnan(mag).sum())

    return {"envelope_spectrum":{"freq":freq,
                                 "mag": mag,
                                 "phase": phase
    }}

def augmentation_signals(mode,severity,results_dict):
    meta_data = results_dict[mode][severity]["meta_data"]
    fs = meta_data["sampling_frequency"]
    expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

    # Using all of the healthy ses as input means that the augmented dataset will have the noise of the training set
    # However, among the augmented dataset the "signal" will be identical
    # notice the "0" meaning that we are using healthy data
    healthy_ses = results_dict[mode]["0"]["envelope_spectrum"]["mag"]# [0] # Use the first one

    ases = AugmentedSES(healthy_ses=healthy_ses,fs=fs,fault_frequency=expected_fault_frequency, peak_magnitude=0.03)
    # print(ases.freqs.shape)

    envelope_specturm = ases.get_augmented_ses()
    return {"augmented_envelope_spectrum":{"freq":ases.freqs,
                                           "mag": envelope_specturm}}

def add_augmented_ses(data_dict):
    # Add features that are derived from the time series simulation
    for mode,dictionary in data_dict.items():
        for severity, measurements in dictionary.items():
            # signal = measurements["time_domain"]
            processed = processed_signals(mode,severity,measurements)
            data_dict[mode][severity].update(processed)

            augmented = augmentation_signals(mode,severity,data_dict)
            data_dict[mode][severity].update(augmented)
    return data_dict




def remove_dc(arr):
    siglen = arr.shape[1]
    use = int(siglen / 4)
    return arr[:, 1:use]

def compute_encodings(data):
    # Loading the dataset
    # Define models
    model_healthy_only = PCA(2)
    model_healthy_only.name = "healthy_only"
    model_healthy_and_augmented = PCA(2)
    model_healthy_and_augmented.name = "healthy_and_augmented"

    # Set up training data
    all_healthy = [data[mode]["0"]["envelope_spectrum"]["mag"] for mode in list(data.keys())]
    healthy_train = remove_dc(
        np.vstack(all_healthy))  # Healthy data from different "modes" even though modes dont technically exist when healthy

    all_augmented_modes = [data[mode]["1"]["augmented_envelope_spectrum"]["mag"] for mode in list(data.keys())]
    augmented_and_healthy_train = remove_dc(np.vstack(all_healthy + all_augmented_modes))

    # Train the models
    model_healthy_only.fit(healthy_train)
    model_healthy_and_augmented.fit(augmented_and_healthy_train)

    # List of trained models
    models = [model_healthy_only, model_healthy_and_augmented]

    for mode_name, mode_data in data.items():
        for severity_name, severity_data in mode_data.items():
            data_type_dict = {}
            for data_type in ["envelope_spectrum", "augmented_envelope_spectrum"]:
                model_type_dict = {}
                for model in models:
                    encoding = model.transform(
                        remove_dc(severity_data[data_type]["mag"]))

                    # Update the dictionary with the encodings
                    model_type_dict.update({model.name: encoding})
                data_type_dict.update({data_type + "_encoding": model_type_dict})
            data[mode_name][severity_name].update(data_type_dict)
    return data


o = PyBearingDatasest(n_severities=10, failure_modes=["ball","inner","outer"], quick_iter=True)
properties_to_modify = {"fault_type":"inner","fault_severity":1}
results_dictionary = o.make_measurements_for_different_failure_mode(properties_to_modify)
results_dictionary = add_augmented_ses(results_dictionary)

# np.save(data_dir.joinpath("generated_and_augmented.npy"),results_dictionary,allow_pickle=True)
# np.save(data_dir.joinpath("generated_and_augmented_rapid_iter.npy"),results_dictionary,allow_pickle=True)

# Add the encoding information

results_dictionary = compute_encodings(results_dictionary)
np.save(data_dir.joinpath("data_with_encodings.npy"), results_dictionary,allow_pickle=True)





