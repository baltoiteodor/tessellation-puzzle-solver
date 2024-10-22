import unittest

import sys

sys.path.append('.')

from solver_tests import TestSolver
from processor_tests import TestProcessor

# Load test_low_num cases from different test_low_num files
testCases1 = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
testCases2 = unittest.TestLoader().loadTestsFromTestCase(TestProcessor)

# Create a test_low_num suite combining all test_low_num cases
testSt = unittest.TestSuite([testCases1, testCases2])

# Run the test_low_num suite
unittest.TextTestRunner().run(testSt)
