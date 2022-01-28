import unittest
from src.data.build_data_and_encodings import run_data_and_encoding_pipeline
from src.data.phenomenological_ses.make_phenomenological_ses import AugmentedSES
from src.data.phenomenological_bearing_model.bearing_model import Measurement
from src.utils.reading_and_writing import get_simulation_properties
from src.utils import sigproc
import numpy as np
from definitions import data_dir



class TestNotebooks(unittest.TestCase):

    def test_analytical_sdof_response(self):
        # import (run) the script to make sure everything runs as intended
        from notebooks import analytical_sdof_response

    def test_squared_envelope_spectrum_sanity_checks(self):
        from notebooks import squared_envelope_spectrum_sanity_checks


class TestBuildDataAndEncodings(unittest.TestCase):

    def test_run_data_and_encoding_pipeline(self):
        run_data_and_encoding_pipeline("data_generated_by_tests", quik_iter=True)

class TestAgmentedSES(unittest.TestCase):

    def test_get_augmented_ses(self):
        results_dictionary = np.load(data_dir.joinpath("data_with_encodings.npy"), allow_pickle=True)[()]
        modes = list(results_dictionary.keys())

        example = results_dictionary[modes[-1]]["0"]["envelope_spectrum"]["mag"]
        print(example.shape)

        aug_obj = AugmentedSES(example, fault_frequency=60, percentage_of_freqs_to_decay_99_percent=0.1)

        # TODO: dedicated dataset for testing
        # TODO: dedicated notebook for experimenting with data augmentation
        # aug_obj.show_augmented_ses()


if __name__ == '__main__':
    unittest.main()
