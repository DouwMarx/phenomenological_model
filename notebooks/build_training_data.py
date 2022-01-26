from src.data.phenomenological_bearing_model.make_data import PyBearingDatasest,envelope,env_spec
import numpy as np
import matplotlib.pyplot as plt
from src.data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from definitions import data_dir

o = PyBearingDatasest(n_severities=2, failure_modes=["ball","inner"])
properties_to_modify = {"fault_type":"inner","fault_severity":1}
results_dictionary = o.make_measurements_for_different_failure_mode(properties_to_modify)


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

    healthy_ses = results_dict[mode][severity]["envelope_spectrum"]["mag"][0] # Use the first one

    ases = AugmentedSES(healthy_ses=healthy_ses,fs=fs,fault_frequency=expected_fault_frequency, peak_magnitude=0.03)

    envelope_specturm = ases.get_augmented_ses()
    return {"augmented_envelope_spectrum":{"freq":ases.freqs,
                                           "mag": envelope_specturm}}


for mode,dictionary in results_dictionary.items():
    print(mode)
    for severity, measurements in dictionary.items():
        # signal = measurements["time_domain"]
        processed = processed_signals(mode,severity,measurements)
        results_dictionary[mode][severity].update(processed)

        augmented = augmentation_signals(mode,severity,results_dictionary)
        results_dictionary[mode][severity].update(augmented)

np.save(data_dir.joinpath("generated_and_augmented.npy"),results_dictionary,allow_pickle=True)



