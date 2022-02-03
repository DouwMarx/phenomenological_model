import unittest



class TestNotebooks(unittest.TestCase):

    def test_analytical_sdof_response(self):
        # Run the script to verity that it is working
        from notebooks.analytical_sdof_response import main
        main()

class TestPhenomenologicalBearingModel(unittest.TestCase):

    def test_make_data(self):
        # Run the script to verity that it is working
        from pypm.phenomenological_bearing_model.make_data import main
        main() # TODO: Make tests more specific

class TestUtils(unittest.TestCase):

    def test_read_phenomenological_data(self):
        # Run the script to verity that it is working
        from pypm.utils.reading_and_writing import main
        d = main() # TODO: Make tests more specific
        assert isinstance(d,dict)


if __name__ == '__main__':
    unittest.main()
