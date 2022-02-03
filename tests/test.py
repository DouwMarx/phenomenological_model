import unittest
import numpy as np
from definitions import data_dir



class TestNotebooks(unittest.TestCase):

    def test_analytical_sdof_response(self):
        # import (run) the script to make sure everything runs as intended
        import notebooks.analytical_sdof_response

    def test_squared_envelope_spectrum_sanity_checks(self):
        # TODO: Move to augmentation
        pass


if __name__ == '__main__':
    unittest.main()
