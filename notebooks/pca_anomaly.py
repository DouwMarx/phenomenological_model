import numpy as np
from sklearn.decomposition import PCA
from definitions import data_dir

# Loading the dataset
data = np.load(data_dir.joinpath("generated_and_augmented.npy"),allow_pickle=True)[()]

