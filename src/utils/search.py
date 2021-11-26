import numpy as np

def find_nearest_index(array, value):
    """Fast search for sorted array: https: // stackoverflow.com / questions / 2566412 / find - nearest - value - in -numpy - array"""
    idx = np.searchsorted(array, value, side="left")
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx
