import unittest

import sys
sys.path.append('.')

from solver_tests import TestSolver
from processor_tests import TestProcessor


# Load test cases from different test files
test_cases_1 = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
test_cases_2 = unittest.TestLoader().loadTestsFromTestCase(TestProcessor)

# Create a test suite combining all test cases
test_suite = unittest.TestSuite([test_cases_1, test_cases_2])

# Run the test suite
unittest.TextTestRunner().run(test_suite)