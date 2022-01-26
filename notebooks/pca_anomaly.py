import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from definitions import data_dir
import plotly.graph_objs as go

def remove_dc(arr):
    siglen = arr.shape[1]
    use = int(siglen/4)
    return arr[:,1:use]

# Loading the dataset
# data = np.load(data_dir.joinpath("generated_and_augmented.npy"),allow_pickle=True)[()]
data = np.load(data_dir.joinpath("generated_and_augmented_rapid_iter.npy"),allow_pickle=True)[()]

# Define models
model_healthy_only = PCA(2)
model_healthy_and_augmented = PCA(2)

# Set up training data
all_healthy = [data[mode]["0"]["envelope_spectrum"]["mag"] for mode in list(data.keys())]
healthy_train = remove_dc(np.vstack(all_healthy)) # Healthy data from different "modes" even though modes dont technically exist when healthy

all_augmented_modes = [data[mode]["1"]["augmented_envelope_spectrum"]["mag"] for mode in list(data.keys())]
augmented_and_healthy_train = remove_dc(np.vstack(all_healthy + all_augmented_modes))

# Set up test data
some_random_mode = "ball"
healthy_test = remove_dc(data[some_random_mode]["0"]["envelope_spectrum"]["mag"])

ball_test = remove_dc(data["ball"]["1"]["envelope_spectrum"]["mag"])
inner_test = remove_dc(data["inner"]["1"]["envelope_spectrum"]["mag"])
outer_test = remove_dc(data["outer"]["1"]["envelope_spectrum"]["mag"])

# Augmented data per failure mode
ball_augment = remove_dc(data["ball"]["1"]["augmented_envelope_spectrum"]["mag"])
inner_augment = remove_dc(data["inner"]["1"]["augmented_envelope_spectrum"]["mag"])
outer_augment = remove_dc(data["outer"]["1"]["augmented_envelope_spectrum"]["mag"])

# Train the models
model_healthy_only.fit(healthy_train)
model_healthy_and_augmented.fit(augmented_and_healthy_train)

encoding_dict_healthy_only = {}
encoding_dict_healthy_and_augmented = {}

modes = ["healthy_test",
         "ball_test","inner_test","outer_test",
         "ball_augment","inner_augment","outer_augment"]

testing_sets = [healthy_test,
                ball_test,inner_test,outer_test,
                ball_augment,inner_augment,outer_augment,
                ]
##


# Do predictions
for mode_test_set, name in zip(testing_sets,modes):
    encoding_dict_healthy_only.update({name:model_healthy_only.transform(mode_test_set)})
    encoding_dict_healthy_and_augmented.update({name:model_healthy_and_augmented.transform(mode_test_set)})



def generate_encoding_plots(encodings_for_mode):
    plt.title()
    fig = go.Figure()
    for mode in modes:
        encoding = encoding_dict_healthy_only[mode]
        fig.add_trace(go.Scatter(x=encoding[:,0], y=encoding[:,0],
                                 mode='markers',
                                 name=mode))

        fig.update_layout(
            title="Model trained on healthy data only",
            xaxis_title="Principle component 1",
            yaxis_title="Principle component 2",
        )

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
