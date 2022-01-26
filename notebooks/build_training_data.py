from src.data.phenomenological_bearing_model.make_data import PyBearingDatasest,envelope,env_spec
import numpy as np
import matplotlib.pyplot as plt
from src.data.phenomenological_ses.make_phenomenological_ses import AugmentedSES

o = PyBearingDatasest(n_severities=2, failure_modes=["ball","inner"])
properties_to_modify = {"fault_type":"inner","fault_severity":1}
results_dictionary = o.make_measurements_for_different_failure_mode(properties_to_modify)



def processed_signals(measurement):
    signal = measurement["time_domain"]
    fs=results_dictionary[mode][severity]["meta_data"]["sampling_frequency"]
    env = envelope(signal)
    # print(np.isnan(env).sum())
    freq, mag, phase = env_spec(signal, fs=fs)
    # print(np.isnan(mag).sum())

    return {"envelope_spectrum":{"freq":freq,
                                 "mag": mag,
                                 "phase": phase
    }}

def augmentation_signals(measurement):
    meta_data = measurement["meta_data"]
    fs = meta_data["sampling_frequency"]
    expected_fault_frequency = meta_data["derived"]["average_fault_frequency"]

    ases = AugmentedSES(healthy_ses=None,fs=fs,fault_frequency=expected_fault_frequency, peak_magnitude=0.03)
    envelope_specturm = ases.get_augmented_ses()
    return {"augmented_envelope_spectrum":envelope_specturm}


    # print(np.isnan(env).sum())

    # return {"Augmented":{"freq":freq,
    #                              "mag": mag,
    #                              "phase": phase
    #                              }}


# plt.figure()

for mode,dictionary in results_dictionary.items():
    print(mode)
    for severity, measurements in dictionary.items():
        # signal = measurements["time_domain"]
        processed = processed_signals(measurements)
        results_dictionary[mode][severity].update(processed)

        augmented = augmentation_signals(measurements)
        results_dictionary[mode][severity].update(augmented)


        plt.figure()
        plt.title("mode:" + mode +" | severity: " + severity)
        meas = results_dictionary[mode][severity]
        real_world = meas["envelope_spectrum"]
        augmented = meas["augmented_envelope_spectrum"]

        plt.plot(real_world["freq"],real_world["mag"][0],label ="Real world")
        plt.plot(real_world["freq"],augmented,label ="Augmented")
        
        plt.legend()


# for mode,dictionary in results_dictionary.items():
#     for severity, measurements in dictionary.items():
#         meta_data = measurements["meta_data"]
#         fault_freq = meta_data["derived"]["average_fault_frequency"]
#
#         plt.vlines(fault_freq,0,0.1, label=mode + " ff")
# plt.legend()




