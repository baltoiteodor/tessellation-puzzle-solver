import unittest

import sys
sys.path.append('.')

from solver_tests import TestSolver
from processor_tests import TestProcessor


# Load test_low_num cases from different test_low_num files
test_cases_1 = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
test_cases_2 = unittest.TestLoader().loadTestsFromTestCase(TestProcessor)

# Create a test_low_num suite combining all test_low_num cases
test_suite = unittest.TestSuite([test_cases_1, test_cases_2])

# Run the test_low_num suite
unittest.TextTestRunner().run(test_suite)