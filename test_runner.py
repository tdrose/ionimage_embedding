import unittest

from tests import test_lightningCLR, test_clr

test_suite = unittest.TestSuite()
loader = unittest.TestLoader()

test_suite.addTests(loader.loadTestsFromModule(test_clr))
test_suite.addTests(loader.loadTestsFromModule(test_lightningCLR))

runner = unittest.TextTestRunner()
result = runner.run(test_suite)

exit(0 if result.wasSuccessful() else 1)