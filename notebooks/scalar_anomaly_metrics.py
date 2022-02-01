import matplotlib.pyplot as plt
import numpy as np
from definitions import data_dir
from scipy.stats import ttest_ind
# seed the random number generator
# generate two independent samples
# compare samples

results_dictionary = np.load(data_dir.joinpath("data_with_encodings.npy"), allow_pickle=True)[()]

# We want to set up a latent traversal model that shows us how the damage is expected to move through the latent space given out augemented dataset


expected_encoding_failure_directions = {}

all_healthy_encoding = [results_dictionary[mode]["0"]["envelope_spectrum_encoding"]["healthy_and_augmented"] for mode in list(results_dictionary.keys())]
all_healthy_encoding = np.vstack(all_healthy_encoding)
healthy_encoding_mean = np.mean(all_healthy_encoding, axis=0)

for mode, dictionary in results_dictionary.items():
    # Making use of a fixed severity for the augmentation for now
    augmented_encoding = results_dictionary[mode]["7"]["augmented_envelope_spectrum_encoding"]["healthy_and_augmented"]
    # healthy_encoding = results_dictionary["ball"]["0"]["envelope_spectrum_encoding"]["healthy_and_augmented"] #TODO: Use all healthy data

    # Currently arbitrarily chosen healthy data
    augmented_encoding_mean = np.mean(augmented_encoding, axis=0)
    # healthy_encoding_mean = np.mean(healthy_encoding, axis=0)

    direction_between_augmented_and_healthy = -(healthy_encoding_mean - augmented_encoding_mean)
    normalized_direction_between_augmented_and_healthy = direction_between_augmented_and_healthy / np.linalg.norm(
        direction_between_augmented_and_healthy)

    expected_encoding_failure_directions.update({mode:direction_between_augmented_and_healthy})


print(expected_encoding_failure_directions)

# Now we fill find the projections of the encodings onto the expected failure directions
# We want to do this for each of the possible failure modes that we are accounting for
for mode, dictionary in results_dictionary.items():
    for severity, measurements in dictionary.items():
        measured_encoding = results_dictionary[mode][severity]["envelope_spectrum_encoding"]["healthy_and_augmented"]

        results_dictionary[mode][severity].update({"measured_projection_in_fault_direction": {},
                                                   "healthy_projection_in_fault_direction": {},
                                                   "hypothesis_test": {},
                                                   })

        # Loop through each of the failure modes that we are accounting for
        for failure_mode, expected_failure_mode_direction in expected_encoding_failure_directions.items():
            measured_projection = np.dot(measured_encoding,expected_failure_mode_direction)

            healthy_encoding = results_dictionary["ball"]["0"]["envelope_spectrum_encoding"]["healthy_and_augmented"]  # TODO: Use all healthy data
            # Currently arbitrarily chosen healthy data
            healthy_projection = np.dot(healthy_encoding, expected_failure_mode_direction)

            results_dictionary[mode][severity]["measured_projection_in_fault_direction"].update({failure_mode: measured_projection})
            results_dictionary[mode][severity]["healthy_projection_in_fault_direction"].update({failure_mode: healthy_projection})

            stat, p = ttest_ind(healthy_projection, measured_projection)
            results_dictionary[mode][severity]["hypothesis_test"].update({failure_mode: p})


# Plot the results

# for mode, dictionary in results_dictionary.items():


def get_stats_with_increasing_severity(true_mode, expected_mode,metric = "hypothesis_test"):
    stats_at_sev = []
    for severity, measurements in results_dictionary[true_mode].items():
        stats_at_sev.append(results_dictionary[true_mode][severity]["hypothesis_test"][expected_mode])

    return np.array(stats_at_sev)

mode = "ball"
expected_mode = "outer"

failure_modes = list(results_dictionary.keys())

for failure_mode in failure_modes:
    plt.figure(failure_mode)

    for expected_mode in failure_modes:
        stats_for_sev = get_stats_with_increasing_severity(failure_mode,expected_mode)
        plt.plot(-np.log(stats_for_sev), label = "expected failure mode: " + expected_mode)

    plt.title("Data from " + failure_mode + " fault mode")
    plt.legend()
    plt.xlabel("severity")
    plt.ylabel("-log(likelihood)")

