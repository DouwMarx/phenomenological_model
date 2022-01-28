import unittest


class TestAnalyticalSDOFResponse(unittest.TestCase):

    def test_analytical_sdof_response(self):
        # import (run) the script to make sure everything runs as intended
        from notebooks import analytical_sdof_response
        # self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")


if __name__ == '__main__':
    unittest.main()
