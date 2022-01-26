import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from definitions import data_dir
import plotly.graph_objs as go


def remove_dc(arr):
    siglen = arr.shape[1]
    use = int(siglen / 4)
    return arr[:, 1:use]


# Loading the dataset
# data = np.load(data_dir.joinpath("generated_and_augmented.npy"),allow_pickle=True)[()]
data = np.load(data_dir.joinpath("generated_and_augmented_rapid_iter.npy"), allow_pickle=True)[()]

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
                model_type_dict.update({model.name: encoding})
            data_type_dict.update({data_type + "_encoding": model_type_dict})
        data[mode_name][severity_name].update(data_type_dict)

        # Update the dictionary with the encodings

# # Set up test data
# some_random_mode = "ball"
# healthy_test = remove_dc(data[some_random_mode]["0"]["envelope_spectrum"]["mag"])
#
# ball_test = remove_dc(data["ball"]["1"]["envelope_spectrum"]["mag"])
# inner_test = remove_dc(data["inner"]["1"]["envelope_spectrum"]["mag"])
# outer_test = remove_dc(data["outer"]["1"]["envelope_spectrum"]["mag"])
#
# # Augmented data per failure mode
# ball_augment = remove_dc(data["ball"]["1"]["augmented_envelope_spectrum"]["mag"])
# inner_augment = remove_dc(data["inner"]["1"]["augmented_envelope_spectrum"]["mag"])
# outer_augment = remove_dc(data["outer"]["1"]["augmented_envelope_spectrum"]["mag"])
#
# encoding_dict_healthy_only = {}
# encoding_dict_healthy_and_augmented = {}
#
# modes = ["healthy_test",
#          "ball_test", "inner_test", "outer_test",
#          "ball_augment", "inner_augment", "outer_augment"]
#
# testing_sets = [healthy_test,
#                 ball_test, inner_test, outer_test,
#                 ball_augment, inner_augment, outer_augment,
#                 ]
# ##
#
#
# # Do predictions
# for mode_test_set, name in zip(testing_sets, modes):
#     encoding_dict_healthy_only.update({name: model_healthy_only.transform(mode_test_set)})
#     encoding_dict_healthy_and_augmented.update({name: model_healthy_and_augmented.transform(mode_test_set)})


# def generate_encoding_plots(data_dict,severity):
#     plt.title()
#     fig = go.Figure()
#     for mode_name,mode_data in data_dict.keys():
#         encoding = encoding_dict_healthy_only[mode]
#         fig.add_trace(go.Scatter(x=encoding[:, 0], y=encoding[:, 0],
#                                  mode='markers',
#                                  name=mode))
#
#         fig.update_layout(
#             title="Model trained on healthy data only",
#             xaxis_title="Principle component 1",
#             yaxis_title="Principle component 2",
#         )

# plt.figure()
# plt.title("Healthy and augmented")
# for mode in modes:
#     encoding = encoding_dict_healthy_and_augmented[mode]
#     plt.scatter(encoding[:,0],encoding[:,1],label=mode)
# plt.legend()
#
#
#
#
