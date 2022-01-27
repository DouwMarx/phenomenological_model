import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from definitions import data_dir
import plotly.graph_objs as go


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

data = np.load(data_dir.joinpath("generated_and_augmented.npy"),allow_pickle=True)[()]
# data = np.load(data_dir.joinpath("generated_and_augmented_rapid_iter.npy"), allow_pickle=True)[()]
data = compute_encodings(data)
np.save(data_dir.joinpath("data_with_encodings.npy"), data,allow_pickle=True)
