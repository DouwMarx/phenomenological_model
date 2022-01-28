import unittest
from src.data.build_data_and_encodings import run_data_and_encoding_pipeline


class TestNotebooks(unittest.TestCase):

    def test_analytical_sdof_response(self):
        # import (run) the script to make sure everything runs as intended
        from notebooks import analytical_sdof_response

    def test_squared_envelope_spectrum_sanity_checks(self):
        from notebooks import squared_envelope_spectrum_sanity_checks


class TestBuildDataAndEncodings(unittest.TestCase):

    def test_run_data_and_encoding_pipeline(self):
        run_data_and_encoding_pipeline("data_generated_by_tests", quik_iter=True)


if __name__ == '__main__':
    unittest.main()
