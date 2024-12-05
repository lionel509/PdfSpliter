import unittest

def run_all_tests():
    """
    Discovers and runs all tests in the 'tests' directory.
    """
    loader = unittest.TestLoader()
    suite = loader.discover('tests')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll tests passed successfully!")
        exit(0)
    else:
        print("\nSome tests failed. Please check the output above for details.")
        exit(1)

if __name__ == "__main__":
    run_all_tests()
